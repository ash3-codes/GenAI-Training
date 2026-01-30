import json
from app.chain import get_ingredient_report

def run_test(name, **kwargs):
    print("\n" + "=" * 90)
    print(f"TEST: {name}")
    print("=" * 90)

    result = get_ingredient_report(**kwargs)

    if result["ok"]:
        print("✅ VALID STRUCTURED OUTPUT\n")
        print(json.dumps(result["data"], indent=2))
    else:
        print("❌ FAILED TO PARSE")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    # TEST RUN 1: OILY + acne
    run_test(
        "Oily skin with acne - Serum,",
        skin_type="",
        skin_needs="",
        concern="",
        product_type=""
    )

    # TEST RUN 2: Dry + pigmentation
    run_test(
        "Dry skin with pigmentation - Moisturiser",
        skin_type="dry",
        skin_needs="tightness, dullness",
        concern="pigmentation",
        product_type="moisturiser",
    )

    # TEST RUN 3: Sensitive scalp + hair fall (Shampoo)
    run_test(
        "Sensitive scalp hair fall - Shampoo",
        skin_type="sensitive",
        skin_needs="sensitivity, irritation",
        concern="hair fall",
        product_type="shampoo",
    )
