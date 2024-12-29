from dataclasses import dataclass
from abc import ABC

#TODO: Add logic for margin account
@dataclass
class CashAccount(ABC):
    initial_capital: float
    """
    Initial capital that can be used for investments.
    """

    def __post_init__(self) -> None:
        self._t: int | None = None
        self._current_capital = self.initial_capital

    @property
    def next_inflow(self) -> float:
        """
        The amount of external capital flowing in (or out) at the next timestep.

        Returns
        -------
        float
           The next inflow (can be negative).
        """
        pass

    @property
    def current_capital(self) -> float:
        return self._current_capital

    def withdraw(self, amount: float):
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
        if amount > self.current_capital:
            raise ValueError(f"Cannot go below 0 account balance.")
        else:
            self._current_capital -= amount
        
    def deposit(self, amount: float):
        """
        Deposits the specified amount into the cash account.

        Parameters
        ----------
        amount : float
            The amount to deposit.
        """
        assert amount >= 0
        self._current_capital += amount

    def reset(self, timestep: int | None = None) -> None:
        """
        Resets the cash account, to be used when resetting the reinforcement learning environment.

        Parameters
        ----------
        timestep : int | None, optional
            The timestep at which to start.
        """
        self._t = timestep or 0
        self._current_capital = self.initial_capital

    def step(self) -> None:
        """
        Evolve the cash account by one timestep, to be used when the reinforcement learning environment takes a step.
        """
        pass
    

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