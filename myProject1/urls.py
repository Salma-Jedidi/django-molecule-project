"""
URL configuration for myProject1 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from myapp import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('myapp/', views.index,name='index'),
    path('Molecule/', views.Mol,name='Mol'),
    path('inscription/', views.inscrip),
    path('test/', views.test, name='test_data'),
    path('inscription/', views.inscrip, name='inscription'),
    path('search/', views.search_results, name='search_results'),
    path('run_app1/', views.run_app1, name='run_app1'),
    path('run_app2/', views.run_app2, name='run_app2'),
    path('run_app3/', views.run_app3, name='run_app3'),
    path('run_app4/', views.run_app4, name='run_app4'),


]
