import os
import re
from typing import Optional, Tuple
import pandas as pd
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer, util
#import google.generativeai as genai

# Merchant hierarchy
MERCHANT_HIERARCHY = {
    # Food & Drink
    "zomato": ("Food & Beverage", "Online Order", "Restaurant", "Delivery"),
    "swiggy": ("Food & Beverage", "Online Order", "Restaurant", "Delivery"),
    "pizza hut": ("Food & Beverage", "Restaurant", "Pizza", None),
    "dominos": ("Food & Beverage", "Quick Service", "Pizza", None),
    "kfc": ("Food & Beverage", "Quick Service", "Fried Chicken", None),
    "starbucks": ("Food & Beverage", "Cafe", "Coffee", "Beverages"),
    "barista": ("Food & Beverage", "Cafe", "Coffee", "Beverages"),
    "mcdonalds": ("Food & Beverage", "Quick Service", "Burgers", None),
    # Shopping / E-commerce
    "amazon purchase": ("Shopping", "Online Retail", "General", "Delivery"),
    "amazon electronics": ("Shopping", "Online Retail", "Electronics", "Delivery"),
    "amazon grocery": ("Groceries", "Online Retail", "Supermarket", "Delivery"),
    "amazon fresh": ("Groceries", "Online Retail", "Supermarket", "Delivery"),
    "flipkart": ("Shopping", "Online Retail", "General", "Delivery"),
    "shoppers stop": ("Shopping", "Fashion & Lifestyle", "Apparel", None),
    "myntra": ("Shopping", "Fashion & Lifestyle", "Apparel", "Delivery"),
    "croma": ("Shopping", "Electronics", "Consumer Tech", None),
    "big basket": ("Groceries", "Online Retail", "Supermarket", "Delivery"),
    "big bazaar": ("Groceries", "Offline Retail", "Supermarket", "In-Store"),
    # Travel / Transport
    "amazon prime": ("Entertainment", "OTT Subscription", "Video", None),
    "uber": ("Travel", "Local Transport", "Taxi/Cab", "App Booking"),
    "olacabs": ("Travel", "Local Transport", "Taxi/Cab", "App Booking"),
    "irctc": ("Travel", "Travel Booking", "Railway", "Online Ticket"),
    "indigo": ("Travel", "Travel Booking", "Airline", "Ticketing"),
    "goibibo": ("Travel", "Travel Booking", "Flights/Hotels", "Online Booking"),
    # Entertainment / Media
    "spotify": ("Entertainment", "Media", "Streaming", "Music"),
    "netflix": ("Entertainment", "Media", "Streaming", "Video"),
    "hotstar": ("Entertainment", "Media", "Streaming", "Video"),
    "zee5": ("Entertainment", "Media", "Streaming", "Video"),
    "bookmyshow": ("Entertainment", "Event Booking", "Movies/Events", "Online Ticket"),
    # Utilities / Bills
    "hp petrol": ("Utilities", "Fuel & Gas", "Petrol", "Station"),
    "reliance gas": ("Utilities", "Fuel & Gas", "LPG", "Cylinder Delivery"),
    "electricity bill": ("Utilities", "Electricity", "Household", "Monthly Payment"),
    "water bill": ("Utilities", "Water", "Household", "Monthly Payment"),
    "jio": ("Utilities", "Telecom/Bill", "Mobile Recharge", None),
    "reliance": ("Utilities", "Telecom/Retail", "Mobile & Gas", "Bill/Recharge"),
    # Healthcare / Pharmacy
    "apollo pharmacy": ("Healthcare", "Pharmacy", "Medicines", "In-Store/Delivery"),
    "medplus": ("Healthcare", "Pharmacy", "Medicines", "In-Store/Delivery"),
    "hospital": ("Healthcare", "Clinical", "Treatment", "Services"),
    # Income / Housing
    "salary credit": ("Income", "Salary/Transfer", "Bank", "Direct Deposit"),
    "rent payment": ("Housing", "Rent", "Apartment", "Monthly"),
}

ALLOWED_CATEGORIES = list({tags[0] for tags in MERCHANT_HIERARCHY.values()})
if "Friends and Family" not in ALLOWED_CATEGORIES:
    ALLOWED_CATEGORIES.append("Friends and Family")

RULE_CONFIDENCE = 0.95
LOW_CONF_LABEL = "Others"

friend_names = [
    "TO RAHUL VERMA", "TO SNEHA VERMA", "IMPS TO", "IMPS FROM"
]

class HybridCategorizer:
    DESC_CANDIDATES = [
        "description", "desc", "narration", "memo", "details",
        "transaction description", "transaction", "remark", "remarks"
    ]

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", name_file_path: Optional[str] = None):
        self.model = SentenceTransformer(model_name)
        self.name_list = set()
        if name_file_path and os.path.exists(name_file_path):
            with open(name_file_path, "r", encoding="utf-8") as f:
                self.name_list = set(line.strip().lower() for line in f if line.strip())
        self.category_embeddings = self.model.encode(ALLOWED_CATEGORIES, convert_to_tensor=True)
        self.api_failed = False

        self.tag_vocabulary = set()
        # Collect all unique subcategory tags from MERCHANT_HIERARCHY
        for tags in MERCHANT_HIERARCHY.values():
            for tag in tags[1:]:
                if tag:
                    self.tag_vocabulary.add(tag)
        self.tag_vocabulary = list(self.tag_vocabulary)
        # Compute embeddings once for tag vocabulary
        self.tag_embeddings = self.model.encode(self.tag_vocabulary, convert_to_tensor=True)


    def preprocess(self, text: Optional[str]) -> str:
        text = (text or "").lower()
        text = re.sub(r"[\/\-\_@]+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def rules_classify(self, description: str) -> Tuple[Optional[str], float, Tuple[Optional[str], Optional[str], Optional[str]]]:
        desc = self.preprocess(description)
        for merchant_key, tags in MERCHANT_HIERARCHY.items():
            merchant_norm = merchant_key.lower()
            if merchant_norm in desc:
                return tags[0], RULE_CONFIDENCE, (tags[1], tags[2], tags[3])
        for merchant_key, tags in MERCHANT_HIERARCHY.items():
            words = merchant_key.split()
            if all(word in desc for word in words if len(word) > 2):
                return tags[0], RULE_CONFIDENCE * 0.9, (tags[1], tags[2], tags[3])
        for merchant_key, tags in MERCHANT_HIERARCHY.items():
            if any(w in desc for w in merchant_key.split()):
                return tags[0], RULE_CONFIDENCE * 0.85, (tags[1], tags[2], tags[3])
        return None, 0.0, (None, None, None)

    def _detect_person_name_dict(self, text: str) -> bool:
        if not text or not self.name_list:
            return False
        tokens = [t.lower() for t in text.split()]
        for i in range(len(tokens) - 1):
            if tokens[i] in self.name_list and tokens[i + 1] in self.name_list:
                return True
        if any(token in self.name_list for token in tokens):
            return True
        return False

    def _match_friend_names(self, description: str) -> bool:
        desc_upper = (description or "").upper()
        for pattern in friend_names:
            if pattern in desc_upper:
                return True
        return False

    # def llm_classify(self, description: str) -> Tuple[str, float, Tuple[Optional[str], Optional[str], Optional[str], str]]:
    #     try:
    #         model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

    #         prompt = (
    #             "You are a finance assistant. Classify the following transaction into one main category "
    #             "and up to three relevant tags describing the transaction.\n"
    #             "Respond strictly in JSON format with keys: category, tag1, tag2, tag3.\n"
    #             "If a tag is not applicable, use null.\n\n"
    #             f"Transaction description: {description}\n\n"
    #             "JSON Response:"
    #         )

    #         response = model.generate_content(
    #             prompt,
    #             generation_config=genai.types.GenerationConfig(
    #                 temperature=0.0,
    #                 max_output_tokens=60,
    #                 top_p=1
    #             )
    #         )

    #         content = response.text.strip()

    #         import json
    #         try:
    #             data = json.loads(content)
    #             category = data.get("category", LOW_CONF_LABEL)
    #             tag1 = data.get("tag1")
    #             tag2 = data.get("tag2")
    #             tag3 = data.get("tag3")
    #             return category, 0.85, (tag1, tag2, tag3, "Gemini LLM")
    #         except json.JSONDecodeError:
    #             # Fallback: treat whole content as category if JSON parsing fails
    #             return content, 0.8, (None, None, None, "Gemini LLM")

    #     except Exception as e:
    #         print(f"LLM API error: {e}")
    #         return LOW_CONF_LABEL, 0.0, (None, None, None, "Gemini LLM Failed")
    def _minilm_tags(self, description: str, max_tags: int = 3, threshold: float = 0.4):
        desc_emb = self.model.encode(description.lower(), convert_to_tensor=True)
        similarities = util.cos_sim(desc_emb, self.tag_embeddings)[0]
        tag_scores = [(self.tag_vocabulary[i], sim.item()) for i, sim in enumerate(similarities) if sim >= threshold]
        tag_scores.sort(key=lambda x: x[1], reverse=True)
        return [tag for tag, _ in tag_scores[:max_tags]]
    
    
    def minilm_classify(self, description: str) -> Tuple[str, float, Tuple[None, None, None]]:
        if not description:
            return LOW_CONF_LABEL, 0.0, (None, None, None)

        if self._match_friend_names(description):
            return LOW_CONF_LABEL, 0.99, (None, None, None)

        if self._detect_person_name_dict(description):
            return "Friends and Family", 0.99, (None, None, None)

        desc_emb = self.model.encode(description.lower(), convert_to_tensor=True)
        similarities = util.cos_sim(desc_emb, self.category_embeddings)
        max_idx = similarities.argmax().item()
        best_cat = ALLOWED_CATEGORIES[max_idx]
        best_score = similarities[0][max_idx].item()

        tags = self._minilm_tags(description)
        tags = (tags + [None]*3)[:3]
        return best_cat, best_score, tuple(tags)

    def categorize(self, description: Optional[str]) -> Tuple[str, Optional[str], Optional[str], Optional[str], str]:
        cat, conf, tags = self.rules_classify(description)
        if cat:
            return cat, tags[0], tags[1], tags[2], "Rule Engine"

        cat, conf, tags = self.minilm_classify(description)
        if cat and cat != LOW_CONF_LABEL:
            return cat, tags[0], tags[1], tags[2], "MiniLM Fallback"

        # cat, conf, tags = self.llm_classify(description)
        # if cat:
        #     if len(tags) == 4:
        #         return cat, tags[0], tags[1], tags[2], tags[3]
        #     return cat, None, None, None, "Gemini LLM"

        return LOW_CONF_LABEL, None, None, None, "Fallback"

    def _find_description_column(self, df: pd.DataFrame, desc_col: Optional[str]) -> Optional[str]:
        if desc_col and desc_col in df.columns:
            return desc_col
        lc_map = {c.lower(): c for c in df.columns}
        for cand in self.DESC_CANDIDATES:
            if cand in lc_map:
                return lc_map[cand]
        for key in ("desc", "narr", "memo", "detail", "remark"):
            for lc, orig in lc_map.items():
                if key in lc:
                    return orig
        return None

    def categorize_df(self, df: pd.DataFrame, desc_col: Optional[str] = "Description") -> pd.DataFrame:
        df = df.copy()
        chosen_col = self._find_description_column(df, desc_col)
        if chosen_col is None:
            raise KeyError(f"No suitable description column found. Columns: {', '.join(df.columns)}")

        df[chosen_col] = df[chosen_col].fillna("").astype(str)
        results = df[chosen_col].apply(lambda x: pd.Series(self.categorize(x)))
        results.columns = ["Category", "Tag_1", "Tag_2", "Tag_3", "Method"]
        return pd.concat([df, results], axis=1)
