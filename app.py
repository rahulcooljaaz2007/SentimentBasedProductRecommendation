from flask import Flask, request, render_template
from model import SentimentBaseProductRecommenderModel

app = Flask(__name__)

sentiment_model = SentimentBaseProductRecommenderModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recomendedProducts', methods=['POST'])
def product_recommendation():
    # Get the username from the HTML form and normalize it
    username = request.form.get('userName', '').strip().lower()

    if not username:
        return render_template("index.html", message="Please enter a valid username.")

    # Retrieve recommended items for the user
    recommended_items = sentiment_model.get_sentiment_based_recommendations(username)

    # Check if recommendations exist
    if recommended_items is not None and not recommended_items.empty:
        print(f"[INFO] Retrieved {len(recommended_items)} items for user '{username}'.")
        print(recommended_items)

        return render_template(
            "index.html",
            column_names=recommended_items.columns.values,
            row_data=recommended_items.values.tolist(),
            zip=zip
        )
    else:
        return render_template(
            "index.html",
            message=f"No product recommendations found for username '{username}'."
        )

if __name__ == '__main__':
    app.run()
