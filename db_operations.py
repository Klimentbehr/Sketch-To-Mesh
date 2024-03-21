from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.errors import DuplicateKeyError
from bson import ObjectId # unsure if needed
import os

from .blender_operations import encode_file, decode_file
from .db_entities import create_file_document, create_user_document

def connect_to_db():
    
    # username: dev | password: dev123
    uri = "mongodb+srv://devdb:dev123@cluster0.aukmt1u.mongodb.net/?retryWrites=true&w=majority"

    client = MongoClient(uri) # creates client from dev db account
    db_name = 'StM-dev'
    db = client[db_name]

    return db 

def test_connection():
    
    db = connect_to_db()

    try:

        # its just going to try to connect and list db collection names
        collection_names = db.list_collection_names()
        print(f"Connected to MongoDB. Collections: {collection_names}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        
        db.client.close()

# saves files to db.
# encodes file in specified file_path parameter, creates a json with the data and inserts into Files collection in the Db
def save_file_to_db(userId, file_path_db, file_name):

    db =  connect_to_db()
    collection = db['Files'] # temporary collection

    # ideally this is going to be file_path_db; the file we want to convert to binary and save in the database. since this is not currently connected to the workflow of the
    # plugin, we left hardcoded to be able to perform a demonstration.
    blend_file_path = file_path_db
    blend_file_name = os.path.basename(blend_file_path)

    file_encoded = encode_file(blend_file_path)

    # the format of data will likely change after login and registration implemented
    # userId, file_name and file_bin_data. userId currently placeholder
    data = create_file_document("123", blend_file_name, file_encoded.getvalue())

    collection.insert_one(data)

    db.client.close()

# TODO: understand if is there a way we can create the client only once. a lot of repetition here, maybe we return the db made already
    
# returns the list of documents that are attached to an userId.
# userId is the unique identifier ObjectID from the user document on the User collections.
# when a new file is saved on the Files Collection, the same objectID from the user document is saved as userId so 
# it can be traced back.
def get_files_by_user_id(userId):
    
    db = connect_to_db()
    collection = db['Files'] # deals with user files being saved. i was thinking of changing this to have a sub-collection for each user
    
    # the return from find is basically a pointer to the query. we need to append to this array so we can properly close the client.
    documents = [] 

    try:
        for document in collection.find({"userId": userId}):
            documents.append(document)
    finally:
        db.client.close()
        
    return documents

# what would it mean to update an document? maybe update a file from the db with a new encoded file?
def update_files_by_object_id(new_data, object_id):
    return 0

def delete_files_by_object_id(object_id_delete):
    db = connect_to_db()
    collection = db['Files']

    # convert the string objectId to ObjectId type
    objectId = ObjectId(object_id_delete)

    result = collection.delete_one({"_id": objectId})

    db.client.close()

    return result.deleted_count # always 0 or 1, because objectId is unique 

# adding to that, I will not implement account deletion here as it would not make sense since we do not have permission levels on the DB.
def insert_user(email, hash_password):
    db = connect_to_db()
    collection = db['Users']
    
    collection.create_index([("username", 1)], unique=True) # userEmail HAS TO BE UNIQUE, and that is a business rule.

    data = create_user_document(email, hash_password)

    try:
        collection.insert_one(data)
        result = 1
    except DuplicateKeyError:  # Catch the DuplicateKeyError
        result = 0
    except Exception as e:
        result = -1
        print(f"LOG-db_operations/insert_user: {e}") # error printed into the console
        

    db.client.close()

    return result

def delete_user(user_id):
    return 0

def update_user(user_id, updated_data):
    return 0

# we access Users collection and return the user document after the query. here we can grab his userId (which is the ObjectID of the file)
# with the objectId, we are able to filter queries in the Files DB so we can return everything this user has inserted.
def get_user_by_email(user_email):
    
    db = connect_to_db()
    collection = db['Users']
    user = []

    documents = collection.find_one({"username": user_email}) # should only return one. business rule
    user.append(documents)

    db.client.close()

    return user


# run the test when this module is executed
if __name__ == "__main__":
    test_connection()