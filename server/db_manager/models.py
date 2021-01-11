from django.db import models


class Visiter(models.Model):
    photo = models.ImageField(upload_to="media/photo")


class CameraMap(models.Model):
    location = models.CharField(max_length=250)
    camera_num = models.CharField(max_length=15)


class DataMap(models.Model):
    visiter = models.ForeignKey(Visiter, on_delete=models.DO_NOTHING)
    camera_map = models.ForeignKey(CameraMap, on_delete=models.DO_NOTHING)
    datetime = models.DateTimeField(auto_now=True)
