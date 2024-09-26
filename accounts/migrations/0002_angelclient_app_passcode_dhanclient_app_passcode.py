# Generated by Django 4.1.10 on 2024-08-01 15:08

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='angelclient',
            name='app_passcode',
            field=models.CharField(default=django.utils.timezone.now, max_length=4),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='dhanclient',
            name='app_passcode',
            field=models.CharField(default=django.utils.timezone.now, max_length=4),
            preserve_default=False,
        ),
    ]
