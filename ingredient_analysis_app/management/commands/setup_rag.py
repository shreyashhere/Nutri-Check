from django.core.management.base import BaseCommand
from django.conf import settings
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ingredient_analysis_app.rag_service import RAGService

class Command(BaseCommand):
    help = 'Setup and initialize the RAG (Retrieval-Augmented Generation) system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--refresh',
            action='store_true',
            help='Refresh the knowledge base even if it already exists',
        )
        parser.add_argument(
            '--docs-path',
            type=str,
            default='./docs',
            help='Path to the documents folder (default: ./docs)',
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🚀 Starting RAG system setup...')
        )

        docs_path = options['docs_path']
        refresh = options['refresh']

        # Check if docs folder exists
        if not os.path.exists(docs_path):
            self.stdout.write(
                self.style.ERROR(f'❌ Documents folder not found: {docs_path}')
            )
            self.stdout.write(
                self.style.WARNING('Please create the docs folder and add your documents.')
            )
            return

        # List documents in the folder
        docs_files = [f for f in os.listdir(docs_path) if f.endswith(('.pdf', '.txt'))]
        if not docs_files:
            self.stdout.write(
                self.style.WARNING('⚠️  No PDF or TXT files found in docs folder')
            )
            return

        self.stdout.write(
            self.style.SUCCESS(f'📚 Found {len(docs_files)} documents:')
        )
        for doc in docs_files:
            self.stdout.write(f'   - {doc}')

        try:
            # Initialize RAG service
            self.stdout.write('🔧 Initializing RAG service...')
            
            if refresh:
                self.stdout.write('🔄 Refreshing knowledge base...')
                rag_service = RAGService(docs_path=docs_path)
                rag_service.refresh_knowledge_base()
            else:
                rag_service = RAGService(docs_path=docs_path)

            if rag_service.documents and len(rag_service.documents) > 0:
                self.stdout.write(
                    self.style.SUCCESS('✅ RAG system initialized successfully!')
                )
                self.stdout.write(
                    self.style.SUCCESS(f'📊 Vector store is ready for queries with {len(rag_service.documents)} documents.')
                )
            else:
                self.stdout.write(
                    self.style.ERROR('❌ Failed to initialize vector store')
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error during RAG setup: {str(e)}')
            )
            return

        self.stdout.write(
            self.style.SUCCESS('🎉 RAG system setup completed!')
        )
        self.stdout.write(
            self.style.SUCCESS('You can now use the enhanced analysis with RAG integration.')
        )
