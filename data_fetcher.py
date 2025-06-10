import requests
import numpy as np
from config import VICTORIA_METRICS_URL, FEATURE_QUERIES, FEATURES

def fetch_latest_data():
    """
    مقدار جدید همه featureها را بر اساس کوئری‌های Prometheus (VictoriaMetrics) برمی‌گرداند.
    Returns: np.ndarray with shape (n_features,)
    """
    feature_values = []
    for feature in FEATURES:
        query = FEATURE_QUERIES[feature]
        url = f"{VICTORIA_METRICS_URL}/api/v1/query"
        params = {"query": query}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # انتظار داریم: data['data']['result'][0]['value'] = [timestamp, value]
            result = data['data']['result']
            if result and 'value' in result[0]:
                value = float(result[0]['value'][1])
            else:
                value = np.nan  # مقدار نامعتبر اگر موجود نبود
        except Exception as e:
            print(f"[data_fetcher] ❌ Error fetching '{feature}': {e}")
            value = np.nan
        feature_values.append(value)
    print("Fetched real data:", feature_values)
    return np.array(feature_values, dtype=np.float32)

# -------------- حالت شبیه‌ساز برای تست و توسعه --------------
def mock_fetch_latest_data():
    """
    برای تست/توسعه (بدون اتصال به دیتا واقعی)، یک بردار تصادفی شبیه داده‌های واقعی تولید می‌کند.
    """
    return np.random.rand(len(FEATURES)).astype(np.float32)

# ------- امکان انتخاب fetcher بر اساس حالت توسعه/واقعی -------
USE_MOCK_DATA = False  # True فقط برای توسعه/تست، False برای دیتای واقعی


def get_data():
    if USE_MOCK_DATA:
        return mock_fetch_latest_data()
    else:
        return fetch_latest_data()






