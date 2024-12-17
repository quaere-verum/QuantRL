from dataclasses import dataclass
from abc import ABC

@dataclass
class CashAccount(ABC):
    initial_capital: float

    def __post_init__(self) -> None:
        self._t: int | None = None
        self._current_capital = self.initial_capital

    @property
    def next_inflow(self) -> float:
        pass

    @property
    def current_capital(self) -> float:
        return self._current_capital

    def withdraw(self, amount: float) -> None:
        pass

    def deposit(self, amount: float) -> None:
        pass

    def reset(self, timestep: int | None = None) -> None:
        self._t = timestep or 0
        self._current_capital = self.initial_capital

    def step(self) -> None:
        pass
    

@dataclass
class ConstantInflowCashAccount(CashAccount):
    inflow_interval: int = 1
    inflow_amount: float = 0

    @property
    def next_inflow(self) -> float:
        return self.inflow_amount if self._t % self.inflow_interval == 0 else 0
    
    def withdraw(self, amount):
        if amount > self.current_capital:
            raise ValueError(f"Cannot go below 0 account balance.")
        else:
            self._current_capital -= amount
        
    def deposit(self, amount):
        assert amount > 0
        self._current_capital += amount
        
    def step(self):
        self._t += 1
        self._current_capital += self.next_inflow