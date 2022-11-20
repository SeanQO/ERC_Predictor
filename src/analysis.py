import project

CSV_PATH = 'src/data/DB-1-CSV.csv'

df = project.load_database(CSV_PATH)
project.print_df(df)
