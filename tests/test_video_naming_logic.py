
import unittest
import pymongo
import datetime
from bson import ObjectId
import os
import re

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')

class TestVideoNaming(unittest.TestCase):
    def setUp(self):
        self.client = pymongo.MongoClient(MONGO_URI)
        self.db = self.client['objectify_db']
        self.users = self.db['users']
        self.test_results = self.db['test_results']
        
        # Create dummy user
        self.user_id = str(ObjectId())
        
        # Insert initial test results
        self.test_results.insert_one({
            'user_id': self.user_id,
            'primary_object': 'saved_video-1',
            'timestamp': datetime.datetime.utcnow()
        })
        self.test_results.insert_one({
            'user_id': self.user_id,
            'primary_object': 'saved_video-3',
            'timestamp': datetime.datetime.utcnow()
        })
        self.test_results.insert_one({
            'user_id': self.user_id,
            'primary_object': 'Unknown', # Should be ignored
            'timestamp': datetime.datetime.utcnow()
        })

    def tearDown(self):
        # Cleanup
        self.test_results.delete_many({'user_id': self.user_id})
        self.client.close()

    def test_naming_logic(self):
        current_user = self.user_id
        
        # LOGIC COPIED FROM APP.PY TO VERIFY
        
        # Check for existing saved_video-N entries
        cursor = self.test_results.find({
            'user_id': current_user,
            'primary_object': {'$regex': '^saved_video-\\d+$'}
        }, {'primary_object': 1})
        
        max_n = 0
        for doc in cursor:
            obj_name = doc.get('primary_object', '')
            try:
                # Extract number
                match = re.search(r'saved_video-(\d+)', obj_name)
                if match:
                    n = int(match.group(1))
                    if n > max_n:
                        max_n = n
            except:
                pass
        
        # Assign next number
        primary_object = f'saved_video-{max_n + 1}'
        
        # Expectation: 3 exists, so next should be 4
        self.assertEqual(primary_object, 'saved_video-4')
        print(f"Test result: {primary_object}")

if __name__ == '__main__':
    unittest.main()
