# -*- coding: utf-8 -*-
"""Tests for the keyless news providers (YFinanceNews + GoogleNewsRSS)."""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import MagicMock, patch

# Mock newspaper before search_service import (optional dependency)
if "newspaper" not in sys.modules:
    mock_np = MagicMock()
    mock_np.Article = MagicMock()
    mock_np.Config = MagicMock()
    sys.modules["newspaper"] = mock_np

from src.search_service import (  # noqa: E402
    GoogleNewsRSSProvider,
    SearchResponse,
    SearchResult,
    SearchService,
    TavilySearchProvider,
    YFinanceNewsProvider,
)


def _iso(hours_ago: float = 1.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rfc822(hours_ago: float = 1.0) -> str:
    return format_datetime(datetime.now(timezone.utc) - timedelta(hours=hours_ago))


def _yf_item(title: str, *, hours_ago: float = 1.0, summary: str = "") -> dict:
    """Current yfinance (>=0.2.5x) shape: payload nested under ``content``."""
    return {
        "id": title,
        "content": {
            "title": title,
            "summary": summary,
            "pubDate": _iso(hours_ago),
            "canonicalUrl": {"url": f"https://finance.yahoo.com/{abs(hash(title))}.html"},
            "provider": {"displayName": "Yahoo Finance"},
        },
    }


def _fake_ticker_factory(items):
    calls = []

    def factory(symbol):
        calls.append(symbol)
        ticker = MagicMock()
        ticker.news = items
        return ticker

    factory.calls = calls
    return factory


RSS_FEED = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>NVDA - Google News</title>
<item>
  <title>Nvidia stock slides as chip selloff deepens - Reuters</title>
  <link>https://news.google.com/rss/articles/abc</link>
  <pubDate>{_rfc822(2)}</pubDate>
  <description>&lt;a href="x"&gt;Nvidia stock slides&lt;/a&gt;&amp;nbsp;&lt;font&gt;Reuters&lt;/font&gt;</description>
  <source url="https://www.reuters.com">Reuters</source>
</item>
<item>
  <title>NVIDIA NVDA earnings preview - Barron's</title>
  <link>https://news.google.com/rss/articles/def</link>
  <pubDate>{_rfc822(5)}</pubDate>
  <description></description>
  <source url="https://www.barrons.com">Barron's</source>
</item>
</channel></rss>"""


class TestYFinanceNewsProvider(unittest.TestCase):
    def test_parses_current_content_shape(self):
        factory = _fake_ticker_factory([_yf_item("Nvidia hits oversold level", summary="<b>RSI</b> at 24")])
        provider = YFinanceNewsProvider(ticker_factory=factory)

        resp = provider.search("ignored", max_results=5, days=3, stock_code="NVDA")

        self.assertTrue(resp.success)
        self.assertEqual(factory.calls, ["NVDA"])
        item = resp.results[0]
        self.assertEqual(item.title, "Nvidia hits oversold level")
        self.assertEqual(item.snippet, "RSI at 24")  # HTML stripped
        self.assertEqual(item.source, "Yahoo Finance")
        self.assertTrue(item.url.startswith("https://finance.yahoo.com/"))
        self.assertIsNotNone(item.retrieved_at)

    def test_parses_legacy_flat_shape(self):
        legacy = {
            "title": "AMD rallies",
            "link": "https://example.com/amd",
            "publisher": "Motley Fool",
            "providerPublishTime": 1_700_000_000,
        }
        provider = YFinanceNewsProvider(ticker_factory=_fake_ticker_factory([legacy]))

        resp = provider.search("q", stock_code="AMD")

        self.assertTrue(resp.success)
        self.assertEqual(resp.results[0].source, "Motley Fool")
        self.assertEqual(resp.results[0].published_date, "1700000000")
        self.assertEqual(resp.results[0].url, "https://example.com/amd")

    def test_respects_max_results_and_skips_untitled(self):
        items = [{"content": {"title": ""}}] + [_yf_item(f"News {i}") for i in range(6)]
        provider = YFinanceNewsProvider(ticker_factory=_fake_ticker_factory(items))

        resp = provider.search("q", max_results=3, stock_code="MSFT")

        self.assertEqual([r.title for r in resp.results], ["News 0", "News 1", "News 2"])

    def test_requires_us_ticker(self):
        factory = _fake_ticker_factory([_yf_item("x")])
        provider = YFinanceNewsProvider(ticker_factory=factory)

        for code in (None, "", "600519", "hk00700", "NVIDIA latest news"):
            resp = provider.search("q", stock_code=code)
            self.assertFalse(resp.success, code)
        self.assertEqual(factory.calls, [])  # never hit the network

    def test_normalize_ticker(self):
        self.assertEqual(YFinanceNewsProvider.normalize_ticker(" nvda "), "NVDA")
        self.assertEqual(YFinanceNewsProvider.normalize_ticker("US.AAPL"), "AAPL")
        self.assertEqual(YFinanceNewsProvider.normalize_ticker("BRK.B"), "BRK.B")
        self.assertIsNone(YFinanceNewsProvider.normalize_ticker("000001"))

    def test_empty_feed_is_unsuccessful(self):
        provider = YFinanceNewsProvider(ticker_factory=_fake_ticker_factory([]))
        resp = provider.search("q", stock_code="GOOGL")
        self.assertFalse(resp.success)
        self.assertIn("GOOGL", resp.error_message)

    def test_exception_becomes_failed_response(self):
        def boom(_symbol):
            raise RuntimeError("yahoo down")

        resp = YFinanceNewsProvider(ticker_factory=boom).search("q", stock_code="META")
        self.assertFalse(resp.success)
        self.assertIn("yahoo down", resp.error_message)

    def test_disabled_provider_is_unavailable(self):
        self.assertFalse(YFinanceNewsProvider(enabled=False).is_available)
        self.assertTrue(YFinanceNewsProvider().is_available)


class TestGoogleNewsRSSProvider(unittest.TestCase):
    @staticmethod
    def _http(status=200, text=RSS_FEED):
        resp = MagicMock()
        resp.status_code = status
        resp.text = text
        return MagicMock(return_value=resp)

    def test_parses_feed_and_strips_source_suffix(self):
        http = self._http()
        provider = GoogleNewsRSSProvider(http_get=http)

        resp = provider.search("NVIDIA NVDA stock latest news", max_results=5, days=3)

        self.assertTrue(resp.success)
        self.assertEqual(
            [r.title for r in resp.results],
            ["Nvidia stock slides as chip selloff deepens", "NVIDIA NVDA earnings preview"],
        )
        self.assertEqual(resp.results[0].source, "Reuters")
        self.assertEqual(resp.results[0].snippet, "Nvidia stock slides Reuters")
        self.assertEqual(resp.results[1].snippet, "NVIDIA NVDA earnings preview")  # falls back to title
        params = http.call_args.kwargs["params"]
        self.assertEqual(params["q"], "NVIDIA NVDA stock latest news when:3d")
        self.assertEqual(params["gl"], "US")

    def test_chinese_locale(self):
        params = GoogleNewsRSSProvider.build_params("英伟达 股票", 0, prefer_chinese=True)
        self.assertEqual(params["q"], "英伟达 股票 when:1d")  # days clamped to >=1
        self.assertEqual(params["hl"], "zh-CN")

    def test_http_error_is_unsuccessful(self):
        resp = GoogleNewsRSSProvider(http_get=self._http(status=429, text="")).search("q")
        self.assertFalse(resp.success)
        self.assertEqual(resp.error_message, "HTTP 429")

    def test_malformed_xml_is_unsuccessful(self):
        resp = GoogleNewsRSSProvider(http_get=self._http(text="<html>captcha")).search("q")
        self.assertFalse(resp.success)

    def test_respects_max_results(self):
        resp = GoogleNewsRSSProvider(http_get=self._http()).search("q", max_results=1)
        self.assertEqual(len(resp.results), 1)


class TestSearchServiceFreeNewsWiring(unittest.TestCase):
    def test_disabled_by_default_in_constructor(self):
        service = SearchService()
        self.assertFalse(service.is_available)

    def test_enabled_appends_keyless_providers_last(self):
        service = SearchService(tavily_keys=["k"], free_news_enabled=True)
        names = [type(p) for p in service._providers]
        self.assertEqual(names, [TavilySearchProvider, YFinanceNewsProvider, GoogleNewsRSSProvider])
        self.assertTrue(service.is_available)
        self.assertTrue(service._constructor_kwargs["free_news_enabled"])

    def _service_with_fakes(self, yf_items, rss_http=None):
        service = SearchService(free_news_enabled=True)
        service._providers = [
            YFinanceNewsProvider(ticker_factory=_fake_ticker_factory(yf_items)),
            GoogleNewsRSSProvider(http_get=rss_http or TestGoogleNewsRSSProvider._http()),
        ]
        return service

    def test_stock_news_uses_yfinance_for_us_ticker(self):
        service = self._service_with_fakes([_yf_item("NVIDIA NVDA stock drops on CEO scandal")])

        resp = service.search_stock_news("NVDA", "NVIDIA Corporation", max_results=3)

        self.assertTrue(resp.success)
        self.assertEqual(resp.provider, "YFinanceNews")
        self.assertIn("NVDA", resp.results[0].title)

    def test_stock_news_falls_back_to_rss_when_yfinance_empty(self):
        service = self._service_with_fakes([])

        resp = service.search_stock_news("NVDA", "NVIDIA Corporation", max_results=3)

        self.assertTrue(resp.success)
        self.assertEqual(resp.provider, "GoogleNewsRSS")

    def test_topic_news_skips_yfinance(self):
        factory = _fake_ticker_factory([_yf_item("should not be used")])
        service = SearchService(free_news_enabled=True)
        service._providers = [
            YFinanceNewsProvider(ticker_factory=factory),
            GoogleNewsRSSProvider(http_get=TestGoogleNewsRSSProvider._http()),
        ]

        service.search_topic_news("semiconductors", focus_keywords=["Nvidia", "chip"])

        self.assertEqual(factory.calls, [])

    @patch("src.search_service.time.sleep", lambda *_: None)
    def test_comprehensive_intel_limits_yfinance_to_latest_news(self):
        factory = _fake_ticker_factory([_yf_item("NVIDIA NVDA latest headline")])
        rss_http = TestGoogleNewsRSSProvider._http()
        service = SearchService(free_news_enabled=True)
        service._providers = [
            YFinanceNewsProvider(ticker_factory=factory),
            GoogleNewsRSSProvider(http_get=rss_http),
        ]

        results = service.search_comprehensive_intel("NVDA", "NVIDIA Corporation", max_searches=3)

        self.assertEqual(results["latest_news"].provider, "YFinanceNews")
        self.assertEqual(factory.calls, ["NVDA"])  # only once, for latest_news
        for dim, resp in results.items():
            if dim != "latest_news":
                self.assertEqual(resp.provider, "GoogleNewsRSS", dim)

    @patch("src.search_service.time.sleep", lambda *_: None)
    def test_comprehensive_intel_keyless_fallback(self):
        rss_http = TestGoogleNewsRSSProvider._http()
        service = SearchService(free_news_enabled=True)
        service._providers = [
            YFinanceNewsProvider(ticker_factory=_fake_ticker_factory([])),
            GoogleNewsRSSProvider(http_get=rss_http),
        ]

        results = service.search_comprehensive_intel("NVDA", "NVIDIA Corporation", max_searches=1)

        self.assertEqual(results["latest_news"].provider, "GoogleNewsRSS")
        self.assertTrue(results["latest_news"].success)


class TestConfigFreeNewsFlag(unittest.TestCase):
    def test_search_capability_counts_free_news(self):
        from src.config import Config

        cfg = Config.__new__(Config)
        for attr in (
            "anspire_api_keys", "bocha_api_keys", "minimax_api_keys",
            "tavily_api_keys", "brave_api_keys", "serpapi_keys", "searxng_base_urls",
        ):
            setattr(cfg, attr, [])
        cfg.searxng_public_instances_enabled = False
        cfg.free_news_enabled = False
        self.assertFalse(cfg.has_search_capability_enabled())
        cfg.free_news_enabled = True
        self.assertTrue(cfg.has_search_capability_enabled())


if __name__ == "__main__":
    unittest.main()
