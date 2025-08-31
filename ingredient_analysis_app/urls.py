from django import urls
from django.urls import path
from . import views
from . import enhanced_views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('', views.home, name='home'),
    path('upload', views.upload, name='upload'),
    path('analyze', views.analyze_ingredients, name='analyze'),
    path('history',views.history, name='history'),
    path('analysis/<int:analysis_id>', views.analysis_detail, name='analysis_detail'),
    
    # RAG-enhanced endpoints
    path('analyze-rag', enhanced_views.analyze_ingredients_with_rag, name='analyze_rag'),
    path('rag-status', enhanced_views.rag_status, name='rag_status'),
    path('refresh-rag', enhanced_views.refresh_rag_knowledge_base, name='refresh_rag'),
    path('ingredient-research/<str:ingredient_name>', enhanced_views.get_ingredient_research, name='ingredient_research'),
    path('rag-dashboard', enhanced_views.rag_dashboard, name='rag_dashboard'),
]