from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
import pickle


class Task(models.Model):
    title = models.CharField('Название', max_length=50)
    task = models.TextField('Описание')

    def __str__(self):
        return self.title

    class Meta:
        verbose_name = 'Задача'
        verbose_name_plural = 'Задача'


class Post(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    date_posted = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.title


class Cluster(models.Model):
    clstr_count = models.IntegerField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name_data = models.CharField(max_length=100)
    k = models.IntegerField()
    n = models.IntegerField()
    lower_bound = models.IntegerField()
    upper_bound = models.IntegerField()
    metrika = models.IntegerField()
    q1 = models.CharField(max_length=100)
    q2 = models.CharField(max_length=100)
    q3 = models.CharField(max_length=100)
    w = models.BinaryField()

    def __str__(self):
        return f'Cluster({self.id})'


