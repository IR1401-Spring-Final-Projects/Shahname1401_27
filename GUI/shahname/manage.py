#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

import torch.nn as nn
import torch.nn.functional as F
class Classifier(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 254)
        self.fc4 = nn.Linear(254, 64)
        self.fc5 = nn.Linear(64, 4)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = F.relu(self.fc4(x))
      x = self.fc5(x)
      return x


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'shahname.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
