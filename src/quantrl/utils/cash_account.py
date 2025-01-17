from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

class FundsError(Exception):
    pass

class AccountType(Enum):
    CASH = 1
    MARGIN = 2
    SHORT = 3

# TODO: Add logic for margin account
@dataclass
class CashAccount(ABC):
    initial_capital: float
    """
    Initial capital that can be used for investments.
    """

    def __post_init__(self) -> None:
        self._t: int | None = None
        self._current_capital = self.initial_capital
        self._current_margin_balance: float | None = None
        self._short_selling_proceeds: float | None = None

    @property
    @abstractmethod
    def next_inflow(self) -> float:
        """
        The amount of external capital flowing into (or out of) 
        the cash account at the next timestep.

        Returns
        -------
        float
           The next inflow (can be negative).
        """
        pass

    def current_balance(self, account: AccountType) -> float:
        match account:
            case AccountType.CASH:
                return self._current_capital
            case AccountType.MARGIN:
                return self._current_margin_balance
            case AccountType.SHORT:
                return self._short_selling_proceeds

    def withdraw(self, amount: float, account: AccountType):
        """
        Withdraws the specified amount of cash.

        Parameters
        ----------
        amount : float
            The amount to withdraw.

        Raises
        ------
        ValueError
            If the amount would bring the account's balance below 0.
        """
        assert amount >= 0
        match account:
            case AccountType.CASH:
                if amount > self._current_capital:
                    raise FundsError("Cannot go below 0 account balance.")
                else:
                    self._current_capital -= amount
            case AccountType.MARGIN:
                if amount > self._current_margin_balance:
                    raise FundsError("Cannot go below 0 margin account balance.")
                else:
                    self._current_margin_balance -= amount
            case AccountType.SHORT:
                if amount > self._short_selling_proceeds:
                    raise FundsError("Cannot go below 0 short selling proceeds.")
                else:
                    self._short_selling_proceeds -= amount
        
    def deposit(self, amount: float, account: AccountType):
        """
        Deposits the specified amount into the cash account.

        Parameters
        ----------
        amount : float
            The amount to deposit.
        """
        assert amount >= 0
        match account:
            case AccountType.CASH:
                self._current_capital += amount
            case AccountType.MARGIN:
                self._current_margin_balance += amount
            case AccountType.SHORT:
                self._short_selling_proceeds += amount

    def reset(self) -> None:
        """
        Resets the cash account, to be used when resetting the reinforcement learning environment.

        Parameters
        ----------
        timestep : int | None, optional
            The timestep at which to start.
        """
        self._t = 0
        self._current_capital = self.initial_capital
        self._current_margin_balance = 0
        self._short_selling_proceeds = 0

    @abstractmethod
    def step(self) -> None:
        """
        Evolve the cash account by one timestep, to be used when the reinforcement learning environment takes a step.
        """
        pass

    def margin_call(self, required_margin: float) -> bool:
        """
        Adjust funds to meet the margin call. Returns True or False based
        on whether or not the margin call was able to be met.

        Parameters
        ----------
        required_margin : float
            The required amount in the margin account.

        Returns
        -------
        bool
            Whether or not it was possible to meet the margin call.
        """
        # TODO: Include collateral from portfolio in margin calculation
        if self._current_margin_balance < required_margin:
            additional_margin = required_margin - self._current_margin_balance
            try:
                self.withdraw(additional_margin, account=AccountType.CASH)
            except FundsError:
                return False
            else:
                self.deposit(additional_margin, account=AccountType.MARGIN)
                return True



@dataclass
class ConstantInflowCashAccount(CashAccount):
    inflow_interval: int = 1
    inflow_amount: float = 0

    @property
    def next_inflow(self) -> float:
        return self.inflow_amount if self._t % self.inflow_interval == 0 else 0
        
    def step(self):
        self._t += 1
        self._current_capital += self.next_inflow