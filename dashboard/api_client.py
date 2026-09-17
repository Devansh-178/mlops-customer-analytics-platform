# dashboard/api_client.py

import requests
from config import API_BASE_URL, REQUEST_TIMEOUT_SECONDS


class APIClientError(Exception):
    """Base exception for all api_client failures."""


class APIUnreachableError(APIClientError):
    """The API couldn't be reached at all — connection refused, DNS failure, timeout."""


class ValidationError(APIClientError):
    """The API rejected the payload with a 422."""
    def __init__(self, detail):
        self.detail = detail
        super().__init__(str(detail))


class ServerError(APIClientError):
    """The API accepted the request but failed internally with a 500."""
    def __init__(self, detail):
        self.detail = detail
        super().__init__(str(detail))


def predict(customer: dict) -> dict:
    """
    POST a single customer record to /predict.
    Returns the parsed PredictionResponse dict on success.
    Raises APIUnreachableError, ValidationError, or ServerError on failure.
    """
    url = f"{API_BASE_URL}/predict"

    try:
        response = requests.post(url, json=customer, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as e:
        raise APIUnreachableError(f"Could not connect to API at {url}") from e
    except requests.exceptions.Timeout as e:
        raise APIUnreachableError(f"API request timed out after {REQUEST_TIMEOUT_SECONDS}s") from e

    if response.status_code == 200:
        return response.json()

    if response.status_code == 422:
        try:
            detail = response.json().get("detail", "Validation failed")
        except ValueError:
            detail = response.text
        raise ValidationError(detail)

    if response.status_code == 500:
        try:
            detail = response.json().get("detail", "Internal server error")
        except ValueError:
            detail = response.text
        raise ServerError(detail)

    response.raise_for_status()
    return response.json()
def predict_batch(records: list[dict]) -> dict:
    """
    POST a list of customer records to /predict/batch.
    Row-level failures are embedded in the response body (status="error"
    per row) rather than raised as exceptions, since one bad row doesn't
    fail the whole request.
    Raises APIUnreachableError or ServerError on transport/server failure.
    """
    url = f"{API_BASE_URL}/predict/batch"

    try:
        response = requests.post(url, json=records, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as e:
        raise APIUnreachableError(f"Could not connect to API at {url}") from e
    except requests.exceptions.Timeout as e:
        raise APIUnreachableError(f"API request timed out after {REQUEST_TIMEOUT_SECONDS}s") from e

    if response.status_code == 200:
        return response.json()

    if response.status_code == 500:
        try:
            detail = response.json().get("detail", "Internal server error")
        except ValueError:
            detail = response.text
        raise ServerError(detail)

    response.raise_for_status()
    return response.json()