import numpy
import matplotlib
from datetime import date
from loan_calculator import Loan
from loan_calculator import display_summary
from loan_calculator import AmortizationScheduleType


def find_the_loan_amount(*, interest_rate, monthly_payment, number_of_months):
    return (
            (monthly_payment / interest_rate) *
            (1 - (1 / (1 + interest_rate) ** number_of_months))
    )


def find_monthly_payment_fixed(*, loan, interest_rate, number_of_months):
    return (loan *
            interest_rate *
            (
                    ((1 + interest_rate) ** number_of_months) /
                    ((1 + interest_rate) ** number_of_months - 1)
            )
            )


def find_monthly_payment_outstanding_balance(*, loan, interest_rate, period_years=10):
    return (loan *
            ((1 + interest_rate) ** period_years - (1 + interest_rate))
            /
            (1)
            )


def find_number_of_months():
    return (
        1
    )


def pkg():
    loan = Loan(
        principal=10000.00,  # principal
        annual_interest_rate=0.05,  # annual interest rate
        start_date=date(2020, 1, 5),  # start date
        return_dates=[
            date(2020, 2, 12),  # expected return date
            date(2020, 3, 13),  # expected return date
            date(2020, 3, 11),  # expected return date
            date(2020, 4, 13),  # expected return date
            date(2020, 5, 12),  # expected return date
            date(2020, 6, 12),  # expected return date
            date(2020, 7, 14),  # expected return date
            date(2020, 8, 15),  # expected return date
        ],
        year_size=365,  # used to convert between annual and daily interest rates
        grace_period=0,  # number of days for which the principal is not affected by the interest rate
        amortization_schedule_type=AmortizationScheduleType.progressive_price_schedule,
        # determines how the principal is amortized
    )

    # print(lc.AmortizationScheduleType.progressive_price_schedule)

    display_summary(loan)



if __name__ == "__main__":
    pkg()

print(find_the_loan_amount(interest_rate=0.01, monthly_payment=10_000, number_of_months=10))
print(find_monthly_payment_fixed(loan=100_000, interest_rate=0.01, number_of_months=12))
