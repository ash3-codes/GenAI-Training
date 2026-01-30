from typing import List
from pydantic import BaseModel, Field

class IngredientReport(BaseModel):
    ingredient_name: str = Field(..., description="Name of the ingredient")
    function: str = Field(..., description="What this ingredient does")
    recommended_products: List[str] = Field(..., description="Product forms it fits into")
    usage_percentage: str = Field(..., description="Typical usage range (string like '0.5% - 2%')")
    safety_notes: str = Field(..., description="Safety and caution notes")
    suitable_for_sensitive_skin: bool = Field(..., description="True/False")

class AvoidIngredient(BaseModel):
    ingredient_name: str = Field(..., description="Ingredient to avoid")
    reason_to_avoid: str = Field(..., description="Why it should be avoided for this user")

class IngredientResponse(BaseModel):
    # Must include section (text-based but included in JSON)
    text_summary: str = Field(..., description="Readable report with recommended ingredients + avoid list")
    recommended_ingredients: List[IngredientReport] = Field(..., description="Minimum 3 recommended ingredients")
    avoid_ingredients: List[AvoidIngredient] = Field(..., description="One or more ingredients to avoid")
