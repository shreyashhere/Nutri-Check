#!/usr/bin/env python3
"""
Simple test script to check RAG functionality
"""
import os
import sys
import logging

# Add the project to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_simple():
    """Test RAG with a simple approach"""
    try:
        # Test if we can import the necessary modules
        logger.info("Testing imports...")
        
        # Test sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("✅ SentenceTransformers imported successfully")
            
            # Test loading a simple model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Model loaded successfully")
            
            # Test embedding
            test_text = "This is a test sentence"
            embedding = model.encode(test_text)
            logger.info(f"✅ Embedding created successfully, shape: {embedding.shape}")
            
        except Exception as e:
            logger.error(f"❌ SentenceTransformers failed: {e}")
            return False
        
        # Test document loading
        try:
            docs_path = os.path.join(os.path.dirname(__file__), "docs")
            if os.path.exists(docs_path):
                files = os.listdir(docs_path)
                logger.info(f"✅ Found {len(files)} files in docs folder")
                for f in files:
                    logger.info(f"   - {f}")
            else:
                logger.error(f"❌ Docs folder not found: {docs_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Document loading failed: {e}")
            return False
        
        logger.info("✅ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_simple()
    if success:
        print("\n🎉 RAG test passed! The system should work.")
    else:
        print("\n❌ RAG test failed. Check the logs above.")
