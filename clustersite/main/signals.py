from django.db.models.signals import pre_save
from django.dispatch import receiver
from .models import Cluster


@receiver(pre_save, sender=Cluster)
def cluster_pre_save(sender, instance, **kwargs):
    # Проверка, что пользователь зарегистрирован или вошел в систему
    if instance.user.is_authenticated:
        # Дополнительная логика, если нужно
        pass
    else:
        # Отменить сохранение объекта Cluster, если пользователь не аутентифицирован
        raise Exception('User must be authenticated to save Cluster object.')