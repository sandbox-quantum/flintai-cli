import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from flintai.eval.common.schema import Message, Content, Role, Part, PartType
from flintai.eval.core.models.model import ModelResponse
from flintai.eval.core.models.model_retry import (
    ExponentialRetryModel,
    FibonacciRetryModel,
    RetryModel,
)


def _make_message():
    return Message(
        content=Content(
            role=Role.USER,
            parts=[Part(part_type=PartType.TEXT, text="hello")],
        ),
    )


def _make_response():
    return ModelResponse(message=None)


class TestExponentialRetryModel(unittest.IsolatedAsyncioTestCase):

    async def test_succeeds_on_first_try(self):
        inner = MagicMock()
        inner.generate = AsyncMock(return_value=_make_response())
        model = ExponentialRetryModel(inner, max_retries=3)
        result = await model.generate(_make_message())
        self.assertEqual(inner.generate.call_count, 1)
        self.assertIsInstance(result, ModelResponse)

    @patch("flintai.eval.core.models.model_retry.random.uniform", return_value=0.0)
    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_transient_error(self, mock_sleep, _):
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=[
            ConnectionError("connection reset"),
            _make_response(),
        ])
        model = ExponentialRetryModel(inner, max_retries=3, base_delay=1.0)
        result = await model.generate(_make_message())
        self.assertEqual(inner.generate.call_count, 2)
        self.assertIsInstance(result, ModelResponse)
        mock_sleep.assert_called_once_with(1.0)

    @patch("flintai.eval.core.models.model_retry.random.uniform", return_value=0.0)
    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_rate_limit(self, mock_sleep, _):
        rate_limit_error = Exception("rate limited")
        rate_limit_error.status_code = 429
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=[
            rate_limit_error,
            rate_limit_error,
            _make_response(),
        ])
        model = ExponentialRetryModel(inner, max_retries=3, base_delay=0.5)
        result = await model.generate(_make_message())
        self.assertEqual(inner.generate.call_count, 3)
        self.assertIsInstance(result, ModelResponse)
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_any_call(0.5)
        mock_sleep.assert_any_call(1.0)

    @patch("flintai.eval.core.models.model_retry.random.uniform", return_value=0.0)
    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_status_attr(self, mock_sleep, _):
        server_error = Exception("503 UNAVAILABLE")
        server_error.status = 503
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=[
            server_error,
            _make_response(),
        ])
        model = ExponentialRetryModel(inner, max_retries=3, base_delay=0.5)
        result = await model.generate(_make_message())
        self.assertEqual(inner.generate.call_count, 2)
        self.assertIsInstance(result, ModelResponse)
        mock_sleep.assert_called_once_with(0.5)

    async def test_raises_non_transient_immediately(self):
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=ValueError("bad input"))
        model = ExponentialRetryModel(inner, max_retries=3)
        with self.assertRaises(ValueError):
            await model.generate(_make_message())
        self.assertEqual(inner.generate.call_count, 1)

    @patch("flintai.eval.core.models.model_retry.random.uniform", return_value=0.0)
    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_raises_after_max_retries(self, mock_sleep, _):
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=TimeoutError("timed out"))
        model = ExponentialRetryModel(inner, max_retries=2)
        with self.assertRaises(TimeoutError):
            await model.generate(_make_message())
        self.assertEqual(inner.generate.call_count, 3)

    @patch("flintai.eval.core.models.model_retry.random.uniform", return_value=0.0)
    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_exponential_backoff(self, mock_sleep, _):
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=[
            OSError("network"),
            OSError("network"),
            OSError("network"),
            _make_response(),
        ])
        model = ExponentialRetryModel(inner, max_retries=3, base_delay=2.0)
        await model.generate(_make_message())
        self.assertEqual(
            mock_sleep.call_args_list,
            [
                unittest.mock.call(2.0),
                unittest.mock.call(4.0),
                unittest.mock.call(8.0),
            ],
        )

    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_jitter_applied(self, mock_sleep):
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=[
            ConnectionError("fail"),
            _make_response(),
        ])
        model = ExponentialRetryModel(inner, max_retries=3, base_delay=1.0)
        await model.generate(_make_message())
        delay = mock_sleep.call_args[0][0]
        self.assertGreaterEqual(delay, 1.0)
        self.assertLessEqual(delay, 2.0)


class TestFibonacciRetryModel(unittest.IsolatedAsyncioTestCase):

    async def test_succeeds_on_first_try(self):
        inner = MagicMock()
        inner.generate = AsyncMock(return_value=_make_response())
        model = FibonacciRetryModel(inner, max_retries=5)
        result = await model.generate(_make_message())
        self.assertEqual(inner.generate.call_count, 1)
        self.assertIsInstance(result, ModelResponse)

    @patch("flintai.eval.core.models.model_retry.random.uniform", return_value=0.5)
    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_transient_error(self, mock_sleep, _):
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=[
            ConnectionError("connection reset"),
            _make_response(),
        ])
        model = FibonacciRetryModel(inner, max_retries=5)
        result = await model.generate(_make_message())
        self.assertEqual(inner.generate.call_count, 2)
        self.assertIsInstance(result, ModelResponse)
        mock_sleep.assert_called_once_with(0.5)

    @patch("flintai.eval.core.models.model_retry.random.uniform", side_effect=lambda a, b: b)
    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_fibonacci_backoff_sequence(self, mock_sleep, _):
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=[
            OSError("network"),
            OSError("network"),
            OSError("network"),
            OSError("network"),
            OSError("network"),
            _make_response(),
        ])
        model = FibonacciRetryModel(inner, max_retries=10)
        await model.generate(_make_message())
        # Fibonacci: 1, 1, 2, 3, 5
        # With jitter returning max, delay equals the fib value
        self.assertEqual(
            mock_sleep.call_args_list,
            [
                unittest.mock.call(1.0),
                unittest.mock.call(1.0),
                unittest.mock.call(2.0),
                unittest.mock.call(3.0),
                unittest.mock.call(5.0),
            ],
        )

    @patch("flintai.eval.core.models.model_retry.random.uniform", side_effect=lambda a, b: b)
    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_delay_capped_at_max(self, mock_sleep, _):
        inner = MagicMock()
        # Need enough retries to reach the cap
        inner.generate = AsyncMock(side_effect=[
            OSError("network"),
        ] * 12 + [_make_response()])
        model = FibonacciRetryModel(
            inner, max_retries=15, max_delay=10.0,
        )
        await model.generate(_make_message())
        delays = [c[0][0] for c in mock_sleep.call_args_list]
        for d in delays:
            self.assertLessEqual(d, 10.0)

    async def test_raises_non_transient_immediately(self):
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=ValueError("bad input"))
        model = FibonacciRetryModel(inner, max_retries=10)
        with self.assertRaises(ValueError):
            await model.generate(_make_message())
        self.assertEqual(inner.generate.call_count, 1)

    @patch("flintai.eval.core.models.model_retry.random.uniform", return_value=0.0)
    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_raises_after_max_retries(self, mock_sleep, _):
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=TimeoutError("timed out"))
        model = FibonacciRetryModel(inner, max_retries=3)
        with self.assertRaises(TimeoutError):
            await model.generate(_make_message())
        self.assertEqual(inner.generate.call_count, 4)

    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_jitter_between_zero_and_fib(self, mock_sleep):
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=[
            ConnectionError("fail"),
            _make_response(),
        ])
        model = FibonacciRetryModel(inner, max_retries=5)
        await model.generate(_make_message())
        delay = mock_sleep.call_args[0][0]
        # First fib value is 1.0, jitter is uniform(0, 1.0)
        self.assertGreaterEqual(delay, 0.0)
        self.assertLessEqual(delay, 1.0)

    @patch("flintai.eval.core.models.model_retry.random.uniform", return_value=0.0)
    @patch("flintai.eval.core.models.model_retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_rate_limit(self, mock_sleep, _):
        rate_limit_error = Exception("rate limited")
        rate_limit_error.status_code = 429
        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=[
            rate_limit_error,
            _make_response(),
        ])
        model = FibonacciRetryModel(inner, max_retries=10)
        result = await model.generate(_make_message())
        self.assertEqual(inner.generate.call_count, 2)
        self.assertIsInstance(result, ModelResponse)


class TestRetryModelAlias(unittest.TestCase):

    def test_retry_model_is_exponential(self):
        self.assertIs(RetryModel, ExponentialRetryModel)


if __name__ == "__main__":
    unittest.main()
