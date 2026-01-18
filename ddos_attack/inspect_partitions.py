from ddos_attack.task import show_partition_distribution

if __name__ == "__main__":
    # Bạn chỉnh theo cấu hình bạn đang chạy
    num_clients = 5
    alpha = 0.5

    show_partition_distribution(
        mode="dirichlet",
        num_partitions=num_clients,
        alpha=alpha,
        seed=42,
        top_k_classes=13,   # bạn có 13 lớp
        plot=True,
    )
