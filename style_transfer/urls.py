from django.conf.urls import url, include
from django.urls import path
from . import views
from django.contrib.auth import login

urlpatterns =[
    #url(r'^w/', include('webapp.urls')),
    url(r'^$', views.index, name='index'),
    url(r'^signup/', views.UserFormView, name='signup'),
    url(r'^login/', views.login_user, name='login'),
    url(r'^main/', views.main_pg, name='main_pg'),
    url(r'^doodle/', views.draw_doodle, name= 'draw_doodle'),
    url(r'^upload/', views.upload_content, name= 'upload_content'),
    url(r'^saved/', views.saved, name= 'saved'),
    url(r'^about_project/', views.about_project, name="about_project"),
	url(r'^about_team/', views.about_team, name="about_team"),
	url(r'^instruction/', views.instruction, name="instruction"),
    #path('main/',views.main_pg,name='main')
]
