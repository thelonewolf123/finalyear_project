# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from django.contrib.auth.models import User


class Online(models.Model):
    user = models.ForeignKey(
        to=User, on_delete=models.CASCADE, null=False, blank=False)
    socketid = models.CharField(max_length=32, blank=False, null=False)
    is_admin = models.BooleanField(blank=False)
