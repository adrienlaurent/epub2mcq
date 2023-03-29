import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

def init_firebase():
    # Fetch the service account key JSON file contents
    cred = credentials.Certificate('*.json')
    # Initialize the app with a service account, granting admin privileges
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://*.firebaseio.com'
    })
