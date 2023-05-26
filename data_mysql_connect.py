import mysql.connector

# Connect to the MySQL database
cnx = mysql.connector.connect(user='xxxx', password='xxxxxx', host='localhost', database='sakila')

# Create a cursor object
cursor = cnx.cursor()

# Define the SQL query
query = "SELECT issue_type, ticket_created_date_time, ticket_closed_date_time FROM 311_data WHERE issue_type = 'Tree Service Emergency'"

# Execute the query
cursor.execute(query)

# Print the results
for (issue_type, ticket_created_date_time, ticket_closed_date_time) in cursor:
    print(issue_type, ticket_created_date_time, ticket_closed_date_time)

# Close the cursor and database connection
cursor.close()
cnx.close()