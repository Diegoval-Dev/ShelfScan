"""
Product categories for ShelfScan detection.
10 categories covering common supermarket shelf segments.
"""

CATEGORIES = {
    0: "bebidas",        # water, soda, juice
    1: "lacteos",        # milk, yogurt, cheese
    2: "snacks",         # chips, crackers, cookies
    3: "cereales",       # breakfast cereals, oats
    4: "limpieza",       # detergent, soap, cleaning products
    5: "enlatados",      # canned goods, tuna, beans
    6: "aceites",        # cooking oils, vinegar, sauces
    7: "higiene",        # shampoo, toothpaste, deodorant
    8: "confiteria",     # candy, chocolate, gum
    9: "zona_vacia",     # empty shelf space (key for stock audit)
}

NUM_CLASSES = len(CATEGORIES)
CLASS_NAMES = list(CATEGORIES.values())

if __name__ == "__main__":
    print(f"Total categories: {NUM_CLASSES}")
    for idx, name in CATEGORIES.items():
        print(f"  {idx:2d}: {name}")
