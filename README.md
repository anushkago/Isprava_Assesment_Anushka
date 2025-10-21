**AI-Based Personal Finance Expense Categorizer**
**Overview**
This project delivers an AI-powered tool for classifying bank transactions into actionable, real-world expense categories—enabling immediate insights from raw statements. Built for Isprava’s Product Manager screening task, the tool combines comprehensive data cleaning, a hybrid semantic rule + AI categorization engine, and modern, interactive dashboard analytics. The solution processes CSVs or bulk-banked data, offers granular tagging, and arms users with visual spending breakdowns for smarter money management.​

**Key Features**
**Robust Transaction Cleaning:** Handles varied formats, parses and corrects narration, ensures all key fields are present.

**Hybrid Categorization Engine:** Combines rule-based merchant logic with AI-powered MiniLM sentence embeddings for semantic similarity—picking the best-fitting category even for unseen descriptions.

**Multi-Tagging System:** Goes beyond primary categories. For each transaction, up to three additional tags are assigned, giving more granular insight (e.g., "Restaurant", "Delivery", "Online Order" for a food-related expense).

**Interactive Streamlit Dashboard:** Allows upload, customization (category list, thresholds), exploration, and export of results in just a few clicks.

**Instant Analytics:** Visualizes spend with bar and pie charts, highlights highest spends, and automatically summarizes patterns.

**Setup & Installation**
**Clone the repository**
bash
git clone https://github.com/<your-username>/AI-Expense-Categorizer.git
cd AI-Expense-Categorizer

**Create and activate a virtual environment**
bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

**Install dependencies**
bash
pip install -r requirements.txt

**Add environment variables**
Create a .env file:
text
GEMINI_API_KEY=your_gemini_or_openrouter_key_here
MINILM_MODEL=all-MiniLM-L6-v2

**Run the dashboard**
bash
streamlit run app.py

**Repository Structure**
<img width="950" height="442" alt="image" src="https://github.com/user-attachments/assets/9b534ef9-b79a-407f-b96a-35481f07a5ee" />


Sample Dataset & Output
Sample-Transactions.csv covers realistic bank entries from various domains (food, travel, shopping, utilities, etc.).

Date	Narration	Debit	Credit	Category	Tag_1	Tag_2	Tag_3	Method
9/24/2025	SWIGGY ORDER 556677@okaxis	380	None	Food & Beverage	Online Order	Restaurant	Delivery	Rule Engine
9/25/2025	UBER TRIP 443322	260	None	Travel	Local Transport	Taxi/Cab	App Booking	Rule Engine
9/23/2025	AMAZON PURCHASE #884433	1499	None	Shopping	Online Retail	General	Delivery	Rule Engine
...	...	...	...	...	...	...	...	...
Example: Multi-Tag Categorization Explained
To provide deeper, more accurate categorization, each transaction can be labeled with up to three tags in addition to the primary category.
Example from dashboard:

Transaction: SWIGGY ORDER 556677@okaxis (2025-09-24)

Result:

Category: Food & Beverage

Tag_1: Online Order

Tag_2: Restaurant

Tag_3: Delivery

Method: Rule Engine

Explanation:
The category "Food & Beverage" is chosen due to the presence of the known merchant "Swiggy," which specializes in restaurant delivery services. To further enrich classification, the system adds three supporting semantic tags:

"Online Order" reflects the platform-based nature.

"Restaurant" clarifies the expense source.

"Delivery" indicates the consumption channel.
This layered tagging gives both the user and analytical models a much clearer picture of spending intent and context, far beyond generic single-label classification.

Explanation of Code Structure
preprocess.py
Reads and normalizes bank data, converts value fields to numeric, strips/cleans narration strings, and eliminates duplicates.

Ensures all essential columns are present for robust downstream processing.

categorize.py
Implements the HybridCategorizer class.

First, checks for direct merchant matches via curated rules (e.g., "Swiggy," "Uber," "IRCTC").

If rules are inconclusive, runs MiniLM sentence embeddings to compute semantic proximity between description and category.

Detects person-to-person transfers and ambiguous descriptors.

Assigns up to three tags using subcategory matching or semantic proximity.

Returns the main category, tags, and method used.

visualize.py
Crafts analytic outputs:

Bar chart: category-wise total debit spend.

Pie chart: proportional spending share per category.

app.py
Host application: handles uploading, preview, cleaning, categorization, analytics, and download steps.

Side panel settings allow adjusting "Rule Confidence" and LLM fallback acceptance.

App Interface & Visuals
Screenshots
​
Upload & dataset preview.

​
Data cleaning and preprocessing.

​
Categorisation of transactions, showing multi-tag output and method.

​
Summary & spending insights section (highest category, totals, top 3).

​
Bar & pie charts for detailed analytics.

User Flow
Upload CSV or use the sample dataset.

Preview cleaned data—auto-standardized and ready for analysis.

Run Categorisation—each transaction is classified, multi-tagged, and method annotated.

Review analytics—major insights and top spenders highlighted.

Visualize spend—chart breakdown for improved financial clarity.

Export results—for records or further use.

Requirements Addressed​
Minimum 30 transactions, 5+ categories (customizable)

CSV/text input, robust cleaning

LLM/NLP-based classification (hybrid, rules + AI)

Multi-tag annotation per transaction

Clear summary (highest category, visualizations)

Functional prototype app (Streamlit dashboard)

Reusable codebase with documented workflow
