from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.say_hello, name='say_hello'),
    path('yt/', views.yt, name='yt'),
    path('llm/', views.llm_answering, name='llm'),
    path('ytvid/', views.process_youtube_video, name='ytvid'),
    path('fileproc/', views.fileproc, name='fileproc'),
    path('answerfile/', views.answerfile, name='answerfile'),
    path('summarize_video/', views.summarize_video, name='summarize_video'),
    path('question_generation/', views.question_generation, name='question_generation'),
]
