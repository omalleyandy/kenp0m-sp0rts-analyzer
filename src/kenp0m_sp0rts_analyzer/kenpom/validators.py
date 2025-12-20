"""Data validation and sanitization for KenPom data.

This module provides validation rules, anomaly detection, and data
sanitization to ensure data quality throughout the pipeline.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    valid: bool
    errors: list[str]
    warnings: list[str]

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def __bool__(self) -> bool:
        return self.valid


@dataclass
class Anomaly:
    """Detected data anomaly."""

    team_id: int
    team_name: str
    field: str
    value: float
    expected_range: tuple[float, float]
    severity: str  # 'warning', 'error'
    message: str


class DataValidator:
    """Validates and sanitizes KenPom data.

    Ensures data quality by checking for:
    - Values within reasonable ranges
    - Required fields present
    - Data consistency
    - Statistical anomalies
    """

    # Valid ranges for key metrics
    VALID_RANGES = {
        # Efficiency metrics
        "AdjEM": (-50.0, 50.0),
        "adj_em": (-50.0, 50.0),
        "AdjOE": (70.0, 150.0),
        "adj_oe": (70.0, 150.0),
        "AdjDE": (70.0, 150.0),
        "adj_de": (70.0, 150.0),
        # Tempo
        "AdjTempo": (55.0, 90.0),
        "adj_tempo": (55.0, 90.0),
        "Tempo": (55.0, 90.0),
        # Luck and strength
        "Luck": (-0.3, 0.3),
        "luck": (-0.3, 0.3),
        "SOS": (-30.0, 30.0),
        "sos": (-30.0, 30.0),
        "Pythag": (0.0, 1.0),
        "pythag": (0.0, 1.0),
        # Four Factors
        "efg_pct_off": (30.0, 70.0),
        "efg_pct_def": (30.0, 70.0),
        "to_pct_off": (5.0, 35.0),
        "to_pct_def": (5.0, 35.0),
        "or_pct_off": (10.0, 50.0),
        "or_pct_def": (10.0, 50.0),
        "ft_rate_off": (10.0, 60.0),
        "ft_rate_def": (10.0, 60.0),
        # Point distribution
        "ft_pct": (5.0, 35.0),
        "two_pct": (25.0, 70.0),
        "three_pct": (10.0, 50.0),
        # Rankings (1-400 for D1 teams)
        "RankAdjEM": (1, 400),
        "rank_adj_em": (1, 400),
        "RankAdjOE": (1, 400),
        "RankAdjDE": (1, 400),
        # Win/Loss
        "Wins": (0, 40),
        "wins": (0, 40),
        "Losses": (0, 40),
        "losses": (0, 40),
    }

    # Required fields for each data type
    REQUIRED_FIELDS = {
        "rating": ["team_id", "team_name", "adj_em", "adj_oe", "adj_de", "adj_tempo"],
        "api_rating": ["TeamName", "AdjEM", "AdjOE", "AdjDE", "AdjTempo"],
        "four_factors": [
            "team_id",
            "efg_pct_off",
            "to_pct_off",
            "or_pct_off",
            "ft_rate_off",
        ],
        "point_dist": ["team_id", "ft_pct", "two_pct", "three_pct"],
        "prediction": [
            "team1_id",
            "team2_id",
            "predicted_margin",
            "predicted_total",
        ],
    }

    # Statistical thresholds for anomaly detection
    ANOMALY_THRESHOLDS = {
        "adj_em": 3.0,  # Standard deviations from mean
        "adj_tempo": 2.5,
        "luck": 2.5,
    }

    def validate_rating(self, rating: dict[str, Any]) -> ValidationResult:
        """Validate a team rating record.

        Args:
            rating: Dictionary with rating data.

        Returns:
            ValidationResult with any errors/warnings.
        """
        errors = []
        warnings = []

        # Check required fields (flexible for both API and internal formats)
        required = (
            self.REQUIRED_FIELDS["api_rating"]
            if "TeamName" in rating
            else self.REQUIRED_FIELDS["rating"]
        )

        for field in required:
            if field not in rating:
                errors.append(f"Missing required field: {field}")
            elif rating[field] is None:
                warnings.append(f"Null value for field: {field}")

        # Check value ranges
        for field, (min_val, max_val) in self.VALID_RANGES.items():
            if field in rating and rating[field] is not None:
                value = rating[field]
                if not min_val <= value <= max_val:
                    if abs(value - min_val) < 5 or abs(value - max_val) < 5:
                        warnings.append(
                            f"{field}={value} near boundary [{min_val}, {max_val}]"
                        )
                    else:
                        errors.append(
                            f"{field}={value} outside valid range [{min_val}, {max_val}]"
                        )

        # Check efficiency consistency
        adj_em = rating.get("AdjEM") or rating.get("adj_em")
        adj_oe = rating.get("AdjOE") or rating.get("adj_oe")
        adj_de = rating.get("AdjDE") or rating.get("adj_de")

        if adj_em is not None and adj_oe is not None and adj_de is not None:
            calculated_em = adj_oe - adj_de
            if abs(calculated_em - adj_em) > 0.5:
                warnings.append(
                    f"AdjEM ({adj_em}) != AdjOE ({adj_oe}) - AdjDE ({adj_de})"
                )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_four_factors(self, data: dict[str, Any]) -> ValidationResult:
        """Validate four factors data.

        Args:
            data: Dictionary with four factors data.

        Returns:
            ValidationResult with any errors/warnings.
        """
        errors = []
        warnings = []

        for field in self.REQUIRED_FIELDS["four_factors"]:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check ranges
        ff_fields = ["efg_pct", "to_pct", "or_pct", "ft_rate"]
        for suffix in ["_off", "_def"]:
            for field in ff_fields:
                full_field = field + suffix
                if full_field in data and data[full_field] is not None:
                    value = data[full_field]
                    min_val, max_val = self.VALID_RANGES.get(
                        full_field, (0, 100)
                    )
                    if not min_val <= value <= max_val:
                        errors.append(
                            f"{full_field}={value} outside [{min_val}, {max_val}]"
                        )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_point_distribution(self, data: dict[str, Any]) -> ValidationResult:
        """Validate point distribution data.

        Args:
            data: Dictionary with point distribution data.

        Returns:
            ValidationResult with any errors/warnings.
        """
        errors = []
        warnings = []

        for field in self.REQUIRED_FIELDS["point_dist"]:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check that percentages sum to ~100%
        ft = data.get("ft_pct", 0) or 0
        two = data.get("two_pct", 0) or 0
        three = data.get("three_pct", 0) or 0
        total = ft + two + three

        if abs(total - 100) > 2:
            warnings.append(
                f"Point percentages sum to {total:.1f}% (expected ~100%)"
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_prediction(self, pred: dict[str, Any]) -> ValidationResult:
        """Validate a game prediction.

        Args:
            pred: Dictionary with prediction data.

        Returns:
            ValidationResult with any errors/warnings.
        """
        errors = []
        warnings = []

        for field in self.REQUIRED_FIELDS["prediction"]:
            if field not in pred:
                errors.append(f"Missing required field: {field}")

        # Check probability range
        win_prob = pred.get("win_probability")
        if win_prob is not None and not 0 <= win_prob <= 1:
            errors.append(f"Invalid win_probability: {win_prob}")

        # Check total is reasonable
        total = pred.get("predicted_total")
        if total is not None and not 80 <= total <= 200:
            warnings.append(f"Unusual predicted_total: {total}")

        # Check confidence interval
        lower = pred.get("confidence_lower")
        upper = pred.get("confidence_upper")
        if lower is not None and upper is not None:
            if lower > upper:
                errors.append(f"Invalid CI: lower ({lower}) > upper ({upper})")
            if upper - lower > 40:
                warnings.append(f"Very wide CI: [{lower}, {upper}]")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def detect_anomalies(
        self,
        ratings: list[dict[str, Any]],
        reference_stats: dict[str, tuple[float, float]] | None = None,
    ) -> list[Anomaly]:
        """Detect statistical anomalies in a set of ratings.

        Args:
            ratings: List of rating dictionaries.
            reference_stats: Optional dict of (mean, std) for each field.
                If not provided, calculates from the data.

        Returns:
            List of detected anomalies.
        """
        import statistics

        anomalies = []

        # Calculate statistics if not provided
        if reference_stats is None:
            reference_stats = {}
            for field in ["adj_em", "adj_tempo", "luck", "AdjEM", "AdjTempo", "Luck"]:
                values = [
                    r[field]
                    for r in ratings
                    if field in r and r[field] is not None
                ]
                if len(values) >= 10:
                    reference_stats[field] = (
                        statistics.mean(values),
                        statistics.stdev(values),
                    )

        # Check each rating against thresholds
        for rating in ratings:
            team_id = rating.get("team_id") or rating.get("TeamID", 0)
            team_name = rating.get("team_name") or rating.get("TeamName", "Unknown")

            for field, threshold in self.ANOMALY_THRESHOLDS.items():
                # Check both naming conventions
                for key in [field, field.replace("_", "").title()]:
                    if key in rating and key in reference_stats:
                        value = rating[key]
                        mean, std = reference_stats[key]

                        if std > 0:
                            z_score = abs(value - mean) / std
                            if z_score > threshold:
                                min_val, max_val = self.VALID_RANGES.get(
                                    key, (mean - 3 * std, mean + 3 * std)
                                )
                                anomalies.append(
                                    Anomaly(
                                        team_id=team_id,
                                        team_name=team_name,
                                        field=key,
                                        value=value,
                                        expected_range=(
                                            mean - threshold * std,
                                            mean + threshold * std,
                                        ),
                                        severity="warning"
                                        if z_score < threshold + 1
                                        else "error",
                                        message=f"{key}={value:.2f} is {z_score:.1f}σ "
                                        f"from mean ({mean:.2f})",
                                    )
                                )

        return anomalies

    def sanitize_response(
        self,
        data: list[dict[str, Any]],
        remove_invalid: bool = False,
    ) -> list[dict[str, Any]]:
        """Sanitize API response data.

        Performs:
        - Type conversion for numeric fields
        - Null handling
        - Optional filtering of invalid records

        Args:
            data: List of response dictionaries.
            remove_invalid: If True, remove records that fail validation.

        Returns:
            Sanitized data list.
        """
        sanitized = []

        for record in data:
            # Convert string numbers to floats
            for field in [
                "AdjEM",
                "AdjOE",
                "AdjDE",
                "AdjTempo",
                "Luck",
                "SOS",
                "Pythag",
            ]:
                if field in record and isinstance(record[field], str):
                    try:
                        record[field] = float(record[field])
                    except (ValueError, TypeError):
                        record[field] = None

            # Convert string integers to ints
            for field in ["Wins", "Losses", "RankAdjEM", "Seed"]:
                if field in record and isinstance(record[field], str):
                    try:
                        record[field] = int(record[field])
                    except (ValueError, TypeError):
                        record[field] = None

            # Validate if filtering is enabled
            if remove_invalid:
                result = self.validate_rating(record)
                if not result.valid:
                    logger.warning(
                        f"Skipping invalid record: {record.get('TeamName')}: "
                        f"{result.errors}"
                    )
                    continue

            sanitized.append(record)

        return sanitized

    def validate_date_range(
        self,
        start_date: date,
        end_date: date,
        min_year: int = 2023,
    ) -> ValidationResult:
        """Validate a date range for archive queries.

        Args:
            start_date: Start of range.
            end_date: End of range.
            min_year: Minimum allowed year (API limitation).

        Returns:
            ValidationResult with any errors/warnings.
        """
        errors = []
        warnings = []

        if start_date > end_date:
            errors.append(f"start_date ({start_date}) > end_date ({end_date})")

        if start_date.year < min_year:
            errors.append(
                f"start_date year ({start_date.year}) < minimum ({min_year})"
            )

        if end_date > date.today():
            warnings.append(f"end_date ({end_date}) is in the future")

        # Check for reasonable range
        days = (end_date - start_date).days
        if days > 365:
            warnings.append(f"Large date range: {days} days")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def check_data_freshness(
        self,
        last_sync: date | None,
        max_age_hours: int = 24,
    ) -> tuple[bool, str]:
        """Check if data is stale and needs refresh.

        Args:
            last_sync: Date of last successful sync.
            max_age_hours: Maximum acceptable age in hours.

        Returns:
            Tuple of (is_fresh, message).
        """
        from datetime import datetime

        if last_sync is None:
            return False, "No sync history found"

        # Convert date to datetime for comparison
        if isinstance(last_sync, date) and not isinstance(last_sync, datetime):
            last_sync = datetime.combine(last_sync, datetime.min.time())

        age = datetime.now() - last_sync
        age_hours = age.total_seconds() / 3600

        if age_hours > max_age_hours:
            return False, f"Data is {age_hours:.1f} hours old (max: {max_age_hours}h)"

        return True, f"Data is {age_hours:.1f} hours old"
