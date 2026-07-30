"""Simple in-memory inventory management for a demo shop."""

from dataclasses import dataclass, field


@dataclass
class Product:
    sku: str
    name: str
    price_cents: int
    quantity: int = 0


class Inventory:
    """Tracks stock levels and applies orders against them."""

    def __init__(self):
        self._products: dict[str, Product] = {}

    def add_product(self, product: Product) -> None:
        self._products[product.sku] = product

    def restock(self, sku: str, amount: int) -> None:
        if sku not in self._products:
            raise KeyError(f"Unknown SKU: {sku}")
        self._products[sku].quantity += amount

    def fulfill_order(self, sku: str, amount: int) -> bool:
        """Attempts to deduct `amount` units of `sku`. Returns False if
        there isn't enough stock, without mutating state."""
        product = self._products.get(sku)
        if product is None or product.quantity < amount:
            return False
        product.quantity -= amount
        return True

    def low_stock(self, threshold: int = 5) -> list[Product]:
        return [p for p in self._products.values() if p.quantity <= threshold]
