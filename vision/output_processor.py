from prettytable import PrettyTable


def output_data(statistics: dict):
    for col_name, stats in statistics.items():
        print(col_name)

        table = PrettyTable()
        table.field_names = ['Name', 'Value']
        for stat_name, value in stats.items():
            table.add_row([stat_name, value])
        table.align = 'l'
        print(table)
