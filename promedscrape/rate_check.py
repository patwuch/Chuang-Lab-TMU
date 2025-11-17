import requests
import time

URL = "https://www.promedmail.org/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RateLimitTestBot/1.0; +https://example.com/bot)"
}

def test_rate_limit(url, max_requests=20, delay=1.0):
    """
    Test how the server responds to repeated requests.

    :param url: URL to test
    :param max_requests: Number of requests to send
    :param delay: Delay in seconds between requests
    """
    print(f"Testing {max_requests} requests to {url} with {delay:.2f}s delay...")

    for i in range(1, max_requests + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            status = r.status_code
            limit_headers = {k: v for k, v in r.headers.items() if "limit" in k.lower() or "retry" in k.lower()}

            print(f"[{i}] Status: {status}", end="")
            if limit_headers:
                print(f" | Rate headers: {limit_headers}")
            else:
                print()

            # Stop if server signals rate-limiting
            if status in (403, 429):
                print("⚠️ Possible rate limit triggered. Stopping test.")
                break

        except requests.RequestException as e:
            print(f"[{i}] Request failed: {e}")
            break

        time.sleep(delay)

# Example: test 20 requests with 1s delay
if __name__ == "__main__":
    test_rate_limit(URL, max_requests=20, delay=1.0)
