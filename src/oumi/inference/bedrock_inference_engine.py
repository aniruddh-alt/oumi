# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from typing import Any, Optional

import boto3
from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.logging import logger

_CONTENT_KEY: str = "content"
_ROLE_KEY: str = "role"


class BedrockInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against the Amazon Bedrock Converse API.

    Notes:
        - Uses boto3's bedrock-runtime client and its `converse` API under the hood.
        - Keeps RemoteInferenceEngine concurrency/retry semantics but swaps transport.
        - v1 scope: text-only requests and responses (no streaming, tools, or images).
    """

    _bedrock_client = None  # Lazily initialized boto3 client

    def _ensure_bedrock_client(self):
        """Lazily import boto3 and create a bedrock-runtime client.

        Credentials and region are resolved by the AWS SDK default chain.
        """
        if self._bedrock_client is not None:
            return self._bedrock_client

        # Let boto3 resolve region from env/config/instance profile
        self._bedrock_client = boto3.client("bedrock-runtime")
        return self._bedrock_client

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to Bedrock Converse API input.

        Returns a dict with keys suitable for boto3.
        """
        # Extract the first SYSTEM message (if any) for top-level system prompt
        system_messages = [
            message for message in conversation.messages if message.role == Role.SYSTEM
        ]
        system_text: Optional[str] = None
        if len(system_messages) > 0:
            # v1: ensure text; coerce non-text content to string
            system_text = (
                system_messages[0].content
                if isinstance(system_messages[0].content, str)
                else str(system_messages[0].content)
            )
            if len(system_messages) > 1:
                logger.warning(
                    """Multiple system messages found; only
                    the first will be used for Bedrock 'system'."""
                )

        # Convert non-system messages to Converse message format
        converse_messages: list[dict[str, Any]] = []
        for message in conversation.messages:
            if message.role == Role.SYSTEM:
                continue
            # v1: only text content supported
            converse_messages.append(
                {
                    "role": message.role.value,
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                message.content
                                if isinstance(message.content, str)
                                else str(message.content)
                            ),
                        }
                    ],
                }
            )

        inference_config: dict[str, Any] = {
            "maxTokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "topP": generation_params.top_p,
        }
        if generation_params.stop_strings:
            inference_config["stopSequences"] = generation_params.stop_strings

        body: dict[str, Any] = {
            "modelId": model_params.model_name,
            "messages": converse_messages,
            "inferenceConfig": inference_config,
        }
        if system_text:
            body["system"] = [{"text": system_text}]

        return body

    @override
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts a Bedrock Converse API response to a conversation."""
        content_blocks = (
            response.get("output", {}).get("message", {}).get("content", [])
        )
        assistant_text: str = ""
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                assistant_text = block.get("text", "")
                break

        new_message = Message(content=assistant_text, role=Role.ASSISTANT)
        return Conversation(
            messages=[*original_conversation.messages, new_message],
            metadata=original_conversation.metadata,
            conversation_id=original_conversation.conversation_id,
        )

    @override
    def _get_request_headers(self, remote_params: RemoteParams) -> dict[str, str]:
        # Not used; boto3 signs requests. Keep empty to satisfy interface.
        return {}

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "max_new_tokens",
            "stop_strings",
            "temperature",
            "top_p",
        }

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=5, politeness_policy=60.0)

    @override
    def _set_required_fields_for_inference(self, remote_params: RemoteParams):
        """Override: Bedrock does not use api_url/api_key; rely on AWS creds.

        Intentionally do nothing to avoid parent's api_url enforcement.
        """
        return

    async def _query_api(
        self,
        conversation: Conversation,
        semaphore,  # PoliteAdaptiveSemaphore or AdaptiveConcurrencyController
        session,  # Unused for Bedrock; kept for compatibility
        inference_config: Optional[Any] = None,
    ) -> Conversation:
        """Invoke Bedrock Converse via boto3 inside concurrency control.

        Mirrors the retry/backoff behavior of RemoteInferenceEngine.
        """
        # Resolve params
        if inference_config is None:
            remote_params = self._remote_params
            generation_params = self._generation_params
            model_params = self._model_params
            output_path = None
        else:
            # InferenceConfig is expected here; keep attribute names aligned
            remote_params = (
                getattr(inference_config, "remote_params", None) or self._remote_params
            )
            generation_params = (
                getattr(inference_config, "generation", None) or self._generation_params
            )
            model_params = (
                getattr(inference_config, "model", None) or self._model_params
            )
            output_path = getattr(inference_config, "output_path", None)

        # Acquire semaphore/adaptive controller
        semaphore_or_controller = (
            self._adaptive_concurrency_controller
            if self._remote_params.use_adaptive_concurrency
            else semaphore
        )

        # Prepare request
        converse_args = self._convert_conversation_to_api_input(
            conversation, generation_params, model_params
        )

        bedrock_client = self._ensure_bedrock_client()

        failure_reason: Optional[str] = None
        async with semaphore_or_controller:
            for attempt in range(remote_params.max_retries + 1):
                try:
                    if attempt > 0:
                        delay = min(
                            remote_params.retry_backoff_base * (2 ** (attempt - 1)),
                            remote_params.retry_backoff_max,
                        )
                        await asyncio.sleep(delay)

                    # boto3 is sync; run in a thread to avoid blocking the loop
                    def _call_converse():
                        return bedrock_client.converse(**converse_args)

                    response = await asyncio.to_thread(_call_converse)

                    try:
                        result = self._convert_api_output_to_conversation(
                            response, conversation
                        )
                        self._save_conversation_to_scratch(result, output_path)
                        await self._try_record_success()
                        return result
                    except Exception as e:  # Parsing/processing error
                        failure_reason = (
                            f"Failed to process successful response: {str(e)}"
                        )
                        await self._try_record_error()
                        if attempt >= remote_params.max_retries:
                            raise RuntimeError(failure_reason) from e
                        continue

                except Exception as e:
                    # Map AWS errors and decide retryability
                    try:
                        from botocore.exceptions import (  # type: ignore
                            BotoCoreError,
                            ClientError,
                        )
                    except Exception:  # pragma: no cover
                        BotoCoreError = Exception  # type: ignore
                        ClientError = Exception  # type: ignore

                    is_client_error = isinstance(e, ClientError)
                    is_core_error = isinstance(e, BotoCoreError)
                    should_retry = False
                    status_code = None

                    if is_client_error:
                        err = getattr(e, "response", {}) or {}
                        status_code = err.get("ResponseMetadata", {}).get(
                            "HTTPStatusCode"
                        )
                        code = err.get("Error", {}).get("Code")
                        # Retry on throttling and 5xx
                        if code in {"ThrottlingException", "TooManyRequestsException"}:
                            should_retry = True
                        if status_code and status_code >= 500:
                            should_retry = True
                    elif is_core_error:
                        # Transient SDK/IO errors are typically retriable
                        should_retry = True

                    failure_reason = f"Bedrock call failed: {str(e)}"
                    await self._try_record_error()

                    if not should_retry or attempt >= remote_params.max_retries:
                        raise RuntimeError(
                            f"Failed to query Bedrock after {attempt + 1} attempts. "
                            + (f"Status {status_code}. " if status_code else "")
                            + failure_reason
                        ) from e
                    continue

        # If loop exits unexpectedly
        raise RuntimeError(
            f"Failed to query Bedrock after {remote_params.max_retries + 1} attempts. "
            + (f"Reason: {failure_reason}" if failure_reason else "")
        )
