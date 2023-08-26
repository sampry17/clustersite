from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name="home"),
    path('about-us', views.about, name='about'),
    path('create', views.create, name='create'),
    path('index', views.index, name='index'),
    path('cluster', views.clustering_view, name='cluster'),
    path('cluster_data', views.cluster_data_view, name='cluster_data'),
    path('recognition_view', views.recognition_view, name='recognition'),
    path('delete-solution/<int:solution_id>/', views.delete_solution, name='delete_solution')
]
