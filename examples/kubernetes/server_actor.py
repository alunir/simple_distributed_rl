from srl.runner.distribution import RabbitMQParameters, RedisParameters, actor_run_forever
from srl.utils import common


def main():
    common.logger_print()
    actor_run_forever(
        RedisParameters(host="redis-internal-service"),
        RabbitMQParameters(host="mq-internal-service", ssl=False),
    )


if __name__ == "__main__":
    main()