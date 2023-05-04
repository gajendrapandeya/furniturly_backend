import os
import psycopg2
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from flask import Flask, request, jsonify

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
    "INSERT INTO user_search_history(user_id, search_query, created_at) VALUES(%s, %s, %s);"
)

INSERT_USERS = (
    "INSERT INTO users(id, email, fullName, mobileNumber, photoUrl) VALUES(%s, %s, %s, %s, %s);"
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


@app.route("/migrate_category")
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


@app.route("/migrate_product")
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

    # Release the cursor and the connection back to the pool
    cur.close()

    return "Products migrated!"


@app.route("/search_history", methods=["POST"])
def save_search_history():
    # Get a connection from the connection pool and create a cursor
    cur = connection.cursor()

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
    query = request.args.get("q")
    sort_by = request.args.get("sort_by", "rating")
    order_by = request.args.get("order_by", "desc")
    price_filter = request.args.get("price_filter")

    # Construct the SQL query based on the parameters
    query_params = []
    sql = "SELECT * FROM products WHERE name LIKE %s"
    query_params.append(f"%{query}%")
    if price_filter:
        price_filter_parts = price_filter.split("-")
        min_price = price_filter_parts[0]
        max_price = price_filter_parts[1]
        sql += " AND price BETWEEN %s AND %s"
        query_params.extend([min_price, max_price])
    if sort_by == "price":
        sql += " ORDER BY price"
    elif sort_by == "rating":
        sql += " ORDER BY rating"
    if order_by == "asc":
        sql += " ASC"
    else:
        sql += " DESC"

    # Execute the SQL query and retrieve the results
    cur.execute(sql, query_params)
    results = cur.fetchall()

    # Convert the results to a list of dictionaries and return as a JSON response
    product_dicts = [dict(row) for row in results]

    # Release the cursor and the connection back to the pool
    cur.close()

    return jsonify(product_dicts)


