import os
import psycopg2
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from flask import Flask, request

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
    connection.close()

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

    # Commit the changes to the database
    connection.commit()

    # Close the database connection
    cur.close()
    connection.close()

    return "Products migrated!"
