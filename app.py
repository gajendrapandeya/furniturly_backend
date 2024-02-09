import math
import os
import re
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import psycopg2.pool
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

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
CORS(app)
url = os.getenv("DATABASE_URL")
connection_pool = psycopg2.pool.SimpleConnectionPool(1, 5, url)

# Initialize Firebase
service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT")
cred = credentials.Certificate("service_account.json")
firebase_admin.initialize_app(cred)


# Function to get connection from pool
def get_db_connection():
    return connection_pool.getconn()


# Function to release connection back to the pool
def release_db_connection(connection):
    connection_pool.putconn(connection)


@app.route("/migrate_category", methods=['POST'])
def migrate_category():
    connection = get_db_connection()
    try:
        cur = connection.cursor()
        cur.execute(CREATE_CATEGORIES_TABLE)
        db = firestore.client()
        categories_ref = db.collection('category')
        categories = categories_ref.get()
        for category in categories:
            category_dict = category.to_dict()
            cur.execute(INSERT_CATEGORY, (category_dict['id'], category_dict['name'], category_dict['imageUrl']))
        connection.commit()
        cur.close()
    finally:
        release_db_connection(connection)
    return "Category Migrated Successfully!"


@app.route("/migrate_product", methods=['POST'])
def migrate_product():
    connection = get_db_connection()
    try:
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
            cur.execute(INSERT_PRODUCT, (
                product['id'], product['categoryId'], product['name'], product['imageUrls'], product['price'],
                product['rating'], product['description'], product['colors']))

        # Commit the changes to the database
        connection.commit()

        # Close the database connection
        cur.close()
    finally:
        release_db_connection(connection)

    return "Products migrated!"


@app.route("/products")
def get_all_products():
    connection = get_db_connection()
    try:
        cur = connection.cursor()
        # Execute the query to select all products
        cur.execute("SELECT * FROM products")

        # Fetch all the rows and convert to a list of dictionaries
        rows = cur.fetchall()
        products = [dict(id=row[0], category_id=row[1], name=row[2], image_urls=row[3], price=row[4], rating=row[5],
                         description=row[6], colors=row[7]) for row in rows]

        connection.commit()
        cur.close()
    finally:
        release_db_connection(connection)

    # Return the list of products as a JSON response
    return jsonify(products)


@app.route("/products/<category_id>")
def get_product_by_category_id(category_id=None):
    connection = get_db_connection()
    try:
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
        products = [dict(id=row[0], category_id=row[1], name=row[2], image_urls=row[3], price=row[4], rating=row[5],
                         description=row[6], colors=row[7]) for row in rows]

        # Close the cursor and the connection back to the pool
        connection.commit()
        cur.close()
    finally:
        release_db_connection(connection)

    # Return the list of products as a JSON response
    return jsonify(products)


@app.route("/migrate_users", methods=['POST'])
def migrate_users():
    connection = get_db_connection()
    try:
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
    finally:
        release_db_connection(connection)

    return "Users Migrated Successfully!"


@app.route("/search_history", methods=["POST"])
def save_search_history():
    connection = get_db_connection()
    try:
        cur = connection.cursor()

        # # Create product table
        cur.execute(CREATE_USER_SEARCH_HISTORY_TABLE)

        # Get the search query and user ID from the POST request
        search_query = request.form.get("search_query")
        user_id = request.form.get("user_id")

        # Save the search history record to the database
        cur.execute(INSERT_USER_SEARCH_HISTORY, (user_id, search_query))
        connection.commit()

        # Release the cursor and the connection back to the pool
        cur.close()
    finally:
        release_db_connection(connection)

    # Return a success message
    return "Search history saved successfully!"


@app.route("/search_products")
def search_products():
    connection = get_db_connection()
    try:
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
    finally:
        release_db_connection(connection)

    return jsonify(products)


# Recommendation Part
def compute_tfidf_matrix(documents):
    # Compute the term frequency (TF) matrix
    tf_matrix = {}
    for doc in documents:
        for word in doc.split():
            if word not in tf_matrix:
                tf_matrix[word] = {}
            if doc not in tf_matrix[word]:
                tf_matrix[word][doc] = 0
            tf_matrix[word][doc] += 1
    for word in tf_matrix:
        for doc in tf_matrix[word]:
            tf_matrix[word][doc] /= len(documents)

    # Compute the inverse document frequency (IDF) vector
    idf_vector = {}
    for doc in documents:
        for word in set(doc.split()):
            if word not in idf_vector:
                idf_vector[word] = 0
            idf_vector[word] += 1
    for word in idf_vector:
        idf_vector[word] = max(1, math.log(len(documents) / (idf_vector[word])))

    # Compute the TF-IDF matrix
    tfidf_matrix = {}
    for doc in documents:
        tfidf_matrix[doc] = {}
        for word in set(doc.split()):
            tfidf_matrix[doc][word] = tf_matrix[word][doc] * idf_vector[word]

    return tfidf_matrix


def compute_cosine_similarity(vec1, vec2):
    dot_product = sum([vec1[word] * vec2.get(word, 0) for word in vec1])
    magnitude1 = math.sqrt(sum([vec1[word] ** 2 for word in vec1]))
    magnitude2 = math.sqrt(sum([vec2[word] ** 2 for word in vec2]))
    return dot_product / (magnitude1 * magnitude2)


@app.route('/recommend/<product_id>', methods=['GET'])
def recommend_products(product_id):
    connection = get_db_connection()
    try:
        # Load the data from the database into a list of dictionaries
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM products')
        rows = cursor.fetchall()
        products = [dict(zip([column[0] for column in cursor.description], row)) for row in rows]

        # Retrieve the product with the given ID
        product = next((p for p in products if p['id'] == product_id), None)

        # Exclude the product from the list of candidates
        candidates = [p for p in products if p['id'] != product_id]

        # Compute the TF-IDF matrix for the candidates
        documents = [p['name'] + ' ' + p['description'] for p in candidates]
        tfidf_matrix = compute_tfidf_matrix(documents)

        # Compute the TF-IDF vector for the product
        product_doc = product['name'] + ' ' + product['description']
        product_tfidf = compute_tfidf_matrix([product_doc])[product_doc]

        # Compute the cosine similarity between the product and the candidates
        similarity_scores = [
            (c, compute_cosine_similarity(product_tfidf, tfidf_matrix[c['name'] + ' ' + c['description']]))
            for c in candidates]

        # Sort the candidates by similarity score and return the top 5
        top_candidates = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:5]

        # Convert the top candidates to a dictionary with camelCase keys
        camelcase_dict = []
        for row in top_candidates:
            # Convert the dictionary keys to camelCase
            camelcase_row = {}
            for key, value in row[0].items():
                camelcase_key = re.sub(r'_([a-z])', lambda m: m.group(1).upper(), key)
                camelcase_row[camelcase_key] = value

            # Convert the price to an integer
            camelcase_row['price'] = int(camelcase_row['price'])

            camelcase_dict.append(camelcase_row)

        # Convert the dictionary to a JSON response and return it
        response = json.dumps(camelcase_dict)
        connection.commit()
        cursor.close()
    finally:
        release_db_connection(connection)
    return Response(response, mimetype='application/json')


# Define a route to retrieve the top trending products
@app.route('/trending-products', methods=['GET'])
def get_trending_products():
    # Load the data from the database into a Pandas DataFrame
    df_products = pd.read_sql_query('SELECT * FROM products', get_db_connection())

    # Compute the popularity of each product
    # It computes the popularity of each product
    # by grouping the products by their id column
    # and counting the number of times each id appears in the DataFrame
    # using the size() function. The resulting DataFrame with the popularity
    # information is stored in a new variable called df_popularity.
    df_popularity = df_products.groupby('id').size().reset_index(name='popularity')

    # Join the popularity data with the product data
    df_ranked_products = pd.merge(df_products, df_popularity, on='id', how='inner')

    # Sort the products by popularity in descending order
    df_ranked_products = df_ranked_products.sort_values(by=['popularity'], ascending=False)

    # Drop the popularity column from the DataFrame
    df_ranked_products = df_ranked_products.drop('popularity', axis=1)

    # Convert the top 10 products to a dictionary with camelCase keys
    top_products = df_ranked_products.head(5)
    camelcase_dict = [{re.sub(r'_([a-z])', lambda m: m.group(1).upper(), k): v for k, v in row.items()} for _, row in
                      top_products.iterrows()]

    # Convert the price to int before sending it back in the response
    for item in camelcase_dict:
        item['price'] = int(item['price'])

    # Convert the dictionary to a JSON response and return it
    response = json.dumps(camelcase_dict)
    return Response(response, mimetype='application/json')


# Recommendation analysis

def evaluate_recommendation_model():
    # Load the data from the database into a list of dictionaries
    cursor = get_db_connection().cursor()
    cursor.execute('SELECT * FROM products')
    rows = cursor.fetchall()
    products = [dict(zip([column[0] for column in cursor.description], row)) for row in rows]

    # Split the data into train and validation sets
    train_data, valid_data = train_test_split(products, test_size=0.2)

    # Compute the TF-IDF matrix for the candidates in the training set
    train_documents = [p['name'] + ' ' + p['description'] for p in train_data]
    train_tfidf_matrix = compute_tfidf_matrix(train_documents)

    # Train the recommendation model
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(train_tfidf_matrix)

    # Evaluate the model on the validation set
    valid_predictions = []
    for product in valid_data:
        product_tfidf = compute_tfidf_matrix([product['name'] + ' ' + product['description']])[
            product['name'] + ' ' + product['description']]
        product_tfidf = np.array(product_tfidf)  # convert to numpy array
        if product_tfidf.size > 1:  # check if there is more than one element
            product_tfidf = product_tfidf.reshape(1, None)
        distances, indices = model.kneighbors(product_tfidf, n_neighbors=5)
        neighbor_indices = indices[0]
        neighbor_products = [train_data[i] for i in neighbor_indices]
        valid_predictions.append([p['id'] for p in neighbor_products])

    # Compute the accuracy of the model
    num_correct = 0
    for i in range(len(valid_data)):
        product_id = valid_data[i]['id']
        predicted_ids = valid_predictions[i]
        if product_id in predicted_ids:
            num_correct += 1
    recommendation_accuracy = num_correct / len(valid_data)

    return recommendation_accuracy


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=os.getenv("PORT"))
