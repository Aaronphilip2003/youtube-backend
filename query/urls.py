from django.urls import path
from . import views

urlpatterns=[
    path('home/', views.say_hello, name='say_hello'),
    path('yt/', views.yt, name='yt'),
    path('llm/', views.llm_answering, name='llm'),
    path('ytvid/', views.process_youtube_video, name='ytvid'),
    path('fileproc/', views.fileproc, name='fileproc'),
]