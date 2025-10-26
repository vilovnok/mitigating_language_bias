import os
import sys
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass, field

from transformers import HfArgumentParser




@dataclass
class ModelArguments:
    """
    Аргументы, относящиеся к тому, какую модель / конфигурацию / токенизатор мы собираемся точно настроить или обучить с нуля.
    """

    model_name_or_path: Optional[str] = field(
        default="bert-base-uncased",
        metadata={"help": "Имя модели или путь к локальной модели HuggingFace."}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Имя конфигурации модели или путь к локальному файлу конфигурации."}
    )
    tokenizer_name: Optional[str] = field(
        default="bert-base-uncased",
        metadata={"help": "Имя токенизатора или путь к локальному токенизатору."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Путь к директории для кеша модели и токенизатора."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Использовать быстрый токенизатор HuggingFace (если доступен)."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Версия модели или ветка репозитория HuggingFace."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Использовать токен авторизации HuggingFace для приватных моделей."}
    )
    use_return_dict: bool = field(
        default=True,
        metadata={"help": "Возвращать выход модели в виде словаря, а не tuple."}
    )

    # Training hyperparameters
    epochs: int = field(
        default=10,
        metadata={"help": "Количество эпох обучения."}
    )
    num_warmup_steps: int = field(
        default=100,
        metadata={"help": "Количество шагов для warmup scheduler."}
    )
    lr: float = field(
        default=3e-5,
        metadata={"help": "Начальный learning rate для оптимизатора."}
    )

    # Contrastive / Pooling parameters
    temp: float = field(
        default=0.05,
        metadata={"help": "Температурный коэффициент для contrastive loss."}
    )
    pooler_type: str = field(
        default="avg_top2",
        metadata={"help": "Тип pooling для получения эмбеддинга: 'cls', 'avg', 'avg_top2' и т.д."}
    )
    hard_negative_weight: float = field(
        default=0.5,
        metadata={"help": "Вес для hard negative примеров (эффективно только если используются hard negatives)."}
    )
    model_hidden_dim: int = field(
        default=768,
        metadata={
            "help": "Размер скрытого слоя основной модели (например, BERT)."
        },
    )
    mlp_hidden_dim: int = field(
        default=512,
        metadata={
            "help": "Размер скрытого слоя MLP, который используется после основной модели."
        },
    )
    mlp_output_dim: int = field(
        default=512,
        metadata={
            "help": "Размер выходного слоя MLP, определяет размер финального эмбеддинга/выхода."
        },
    )




@dataclass
class DataTrainingArguments:
    """
    Аргументы, относящиеся к тому, какие данные мы собираемся ввести в нашу модель для обучения и оценки.
    """

    dataset_name: Optional[str] = field(
        default="nli_for_simcse.csv",
        metadata={"help": "Имя датасета для использования (через библиотеку datasets)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Имя конфигурации датасета для использования (через библиотеку datasets)."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Перезаписывать кешированные версии train/eval датасетов."}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "Процент обучающего датасета, используемый как validation, если нет отдельного split."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "Количество процессов для предобработки данных."}
    )

    # Input / Tokenization
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Файл обучающих данных (.txt или .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "Максимальная длина входной последовательности после токенизации. Длинные последовательности будут усечены."
        }
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Паддинг всех примеров до max_seq_length. Если False, паддинг делается динамически в батче."
        }
    )
    batch_size: int = field(
        default=64,
        metadata={"help": "Размер батча для обучения или инференса."}
    )
    num_sent: int = field(
        default=3,
        metadata={"help": "Количество предложений, используемых для contrastive loss (например, anchor + positives + negatives)."}
    )

    # Output / Task-specific
    jd_num_classes: int = field(
        default=2,
        metadata={"help": "Количество классов для задачи классификации JD."}
    )
    jf_num_classes: int = field(
        default=10,
        metadata={"help": "Размер выходного слоя или количество классов для задачи JF (например, эмбеддинг)."}
    )


def get_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses(args=[])
    return model_args, data_args