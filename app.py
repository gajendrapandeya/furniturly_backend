import os

import psycopg2
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import pandas as pd
import json
import re

CREATE_USERS_TABLE = (
    "CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, email TEXT, fullName TEXT, mobileNumber TEXT, photoUrl TEXT);"
)

CREATE_CATEGORIES_TABLE = (
    "CREATE TABLE IF NOT EXISTS categories(id TEXT PRIMARY KEY, name TEXT, imageUrl TEXT);"
)

CREATE_PRODUCT_TABLE = (
    """CREATE TABLE IF NOT EXISTS products (
     id TEXT PRIMARY KEY,
     category_id TEXT REFERENCES categories(id),
     name TEXT,
     image_urls TEXT[],
     price DOUBLE PRECISION,
     rating DOUBLE PRECISION,
     description TEXT,
     colors TEXT[]
 );"""
)

CREATE_USER_SEARCH_HISTORY_TABLE = (
    """CREATE TABLE IF NOT EXISTS user_search_history (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    search_query TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
""")

INSERT_USER_SEARCH_HISTORY = (
    "INSERT INTO user_search_history(user_id, search_query) VALUES(%s, %s);"
)

INSERT_USERS = (
    "INSERT INTO users(id, email, fullName, mobileNumber, photoUrl) VALUES(%s, %s, %s, COALESCE(%s, NULL), %s);"
)

INSERT_CATEGORY = (
    "INSERT INTO categories(id, name, imageUrl) VALUES(%s, %s, %s);"
)

INSERT_PRODUCT = (
    "INSERT INTO products(id, category_id, name, image_urls, price, rating, description, colors) VALUES(%s, %s, %s, %s, %s, %s, %s, %s);"
)

load_dotenv()

app = Flask(__name__)
url = os.getenv("DATABASE_URL")
service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT")
connection = psycopg2.connect(url)
# Initialize the Firebase app with service account credentials
cred = credentials.Certificate("service_account.json")
firebase_admin.initialize_app(cred)

CORS(app)


@app.route("/migrate_category", methods=['POST'])
def migrate_category():
    cur = connection.cursor()

    # Create the necessary tables in the database if they don't already exist
    cur.execute(CREATE_CATEGORIES_TABLE)

    # Get the data from the Firebase collections
    db = firestore.client()
    categories_ref = db.collection('category')

    categories = categories_ref.get()

    # Insert the data retrieved from Firebase into the corresponding tables in the PostgresSQL database
    for category in categories:
        category = category.to_dict()
        cur.execute(INSERT_CATEGORY, (category['id'], category['name'], category['imageUrl']))

    # Commit the changes to the database
    connection.commit()

    # Close the database connection
    cur.close()

    return "Category Migrated Successfully!"


@app.route("/migrate_product", methods=['POST'])
def migrate_product():
    cur = connection.cursor()

    # Create product table
    cur.execute(CREATE_PRODUCT_TABLE)

    # Get the data from the Firebase collections
    db = firestore.client()
    products_ref = db.collection('product')

    products = products_ref.get()

    # Insert the data retrieved from Firebase into the corresponding tables in the PostgresSQL database
    for product in products:
        product = product.to_dict()
        print(product['categoryId'])
        cur.execute(INSERT_PRODUCT, (
            product['id'], product['categoryId'], product['name'], product['imageUrls'], product['price'],
            product['rating'], product['description'], product['colors']))

    # Commit the changes to the database
    connection.commit()

    # Close the database connection
    cur.close()

    return "Products migrated!"


@app.route("/products")
def get_all_products():
    cur = connection.cursor()

    # Execute the query to select all products
    cur.execute("SELECT * FROM products")

    # Fetch all the rows and convert to a list of dictionaries
    rows = cur.fetchall()
    print(rows)
    products = [dict(id=row[0], category_id=row[1], name=row[2], image_urls=row[3], price=row[4], rating=row[5],
                     description=row[6], colors=row[7]) for row in rows]

    # Close the cursor and the connection back to the pool
    cur.close()

    # Return the list of products as a JSON response
    return jsonify(products)


@app.route("/products/<category_id>")
def get_product_by_category_id(category_id=None):
    cur = connection.cursor()

    # Check if a category_id is provided
    if category_id:
        # Execute the query to select products by category_id
        cur.execute("SELECT * FROM products WHERE category_id = %s", (category_id,))
    else:
        # Execute the query to select all products
        cur.execute("SELECT * FROM products")

    # Fetch all the rows and convert to a list of dictionaries
    rows = cur.fetchall()
    print(rows)
    products = [dict(id=row[0], category_id=row[1], name=row[2], image_urls=row[3], price=row[4], rating=row[5],
                     description=row[6], colors=row[7]) for row in rows]

    # Close the cursor and the connection back to the pool
    cur.close()

    # Return the list of products as a JSON response
    return jsonify(products)


@app.route("/migrate_users", methods=['POST'])
def migrate_users():
    cur = connection.cursor()

    # Create the necessary tables in the database if they don't already exist
    cur.execute(CREATE_USERS_TABLE)

    # Get the data from the Firebase collections
    db = firestore.client()
    users_ref = db.collection('users')

    users = users_ref.get()

    # Insert the data retrieved from Firebase into the corresponding tables in the PostgresSQL database
    for user in users:
        user = user.to_dict()
        cur.execute(INSERT_USERS,
                    (user['id'], user['email'], user['fullName'], user.get('mobileNumber'), user['photoUrl']))

    # Commit the changes to the database
    connection.commit()

    # Close the database connection
    cur.close()

    return "Users Migrated Successfully!"


@app.route("/search_history", methods=["POST"])
def save_search_history():
    # Get a connection from the connection pool and create a cursor
    cur = connection.cursor()

    # # Create product table
    cur.execute(CREATE_USER_SEARCH_HISTORY_TABLE)

    # Get the search query and user ID from the POST request
    search_query = request.form.get("search_query")
    user_id = request.form.get("user_id")

    print(search_query)
    print(user_id)

    # Save the search history record to the database
    cur.execute(INSERT_USER_SEARCH_HISTORY, (user_id, search_query))
    connection.commit()

    # Release the cursor and the connection back to the pool
    cur.close()

    # Return a success message
    return "Search history saved successfully!"


@app.route("/search_products")
def search_products():
    # Get a connection from the connection pool and create a cursor
    cur = connection.cursor()

    # Get the search query and filter/sort parameters from the query string
    product_name = request.args.get("product_name")
    price_filter = request.args.get("price_filter")
    filter_param = request.args.get("filter")

    # Construct the SQL query based on the parameters
    query_params = []
    sql = "SELECT * FROM products WHERE 1 = 1"
    if product_name:
        sql += " AND name ILIKE %s"
        query_params.append(f"%{product_name}%")
    if price_filter:
        price_filter_parts = price_filter.split("-")
        min_price = price_filter_parts[0]
        max_price = price_filter_parts[1]
        sql += " AND price BETWEEN %s AND %s"
        query_params.extend([min_price, max_price])
    if filter_param == "LowToHigh":
        sql += " ORDER BY price ASC"
    elif filter_param == "HighToLow":
        sql += " ORDER BY price DESC"
    elif filter_param == "Rating":
        sql += " ORDER BY rating DESC"

    # Execute the SQL query and retrieve the results
    cur.execute(sql, query_params)
    rows = cur.fetchall()

    # Convert the results to a list of dictionaries and return as a JSON response
    products = [
        dict(id=row[0], category_id=row[1], name=row[2], image_urls=row[3], price=row[4], rating=row[5],
             description=row[6], colors=row[7]) for row in rows]

    # Release the cursor and the connection back to the pool
    cur.close()

    return jsonify(products)


@app.route('/recommend/<product_id>', methods=['GET'])
def recommend_products(product_id):
    # Load the data from the database into a Pandas DataFrame
    df_products = pd.read_sql_query('SELECT * FROM products', connection)

    # Retrieve the product with the given ID
    product = df_products[df_products['id'] == product_id].iloc[0]

    # Exclude the product from the list of candidates
    df_candidates = df_products[df_products['id'] != product_id]

    # Compute the TF-IDF vector for the product's name and description
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_candidates['name'] + ' ' + df_candidates['description'])

    # Compute the cosine similarity between the product and the candidates
    product_tfidf = tfidf_vectorizer.transform([product['name'] + ' ' + product['description']])
    similarity_scores = cosine_similarity(product_tfidf, tfidf_matrix)[0]

    # Sort the candidates by similarity score and return the top 5
    top_indices = similarity_scores.argsort()[::-1][:5]
    top_candidates = df_candidates.iloc[top_indices]

    # Convert the top candidates to a dictionary with camelCase keys
    camelcase_dict = [{re.sub(r'_([a-z])', lambda m: m.group(1).upper(), k): v for k, v in row.items()} for _, row in
                      top_candidates.iterrows()]

    # Convert the dictionary to a JSON response and return it
    response = json.dumps(camelcase_dict)
    return Response(response, mimetype='application/json')
