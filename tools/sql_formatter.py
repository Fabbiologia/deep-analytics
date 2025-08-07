import re

def format_sql_for_humans(sql):
    """
    Format a SQL query to be more readable for humans.
    
    Args:
        sql (str): The SQL query to format
        
    Returns:
        str: A formatted version of the SQL query
    """
    if not sql or not isinstance(sql, str):
        return str(sql)
        
    # Strip extra whitespace
    sql = sql.strip()
    
    # Replace SQL keywords with uppercase versions
    keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 
                'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'LIMIT',
                'DISTINCT', 'AND', 'OR', 'AS', 'ON', 'SUM', 'AVG', 'COUNT', 'MAX', 'MIN']
    
    # Format the SQL with newlines
    formatted_sql = sql
    for keyword in keywords:
        # Handle different case variations
        pattern = re.compile(r'\b' + keyword + r'\b', re.IGNORECASE)
        replacement = '\n' + keyword.upper()
        formatted_sql = pattern.sub(replacement, formatted_sql)
    
    # Add indentation for better readability
    lines = formatted_sql.split('\n')
    indented_lines = []
    indent_level = 0
    
    for line in lines:
        if line.strip():  # Skip empty lines
            # Decrease indent for certain keywords
            if any(line.strip().upper().startswith(k) for k in ['FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING']):
                indent_level = max(0, indent_level - 1)
            
            # Add the line with proper indentation
            indented_lines.append('  ' * indent_level + line.strip())
            
            # Increase indent after certain keywords
            if any(line.strip().upper().startswith(k) for k in ['SELECT']):
                indent_level += 1
    
    # Join lines and return
    return '\n'.join(indented_lines)

def translate_sql_to_english(sql):
    """
    Translate SQL query to a plain English description.
    
    Args:
        sql (str): The SQL query to translate
        
    Returns:
        str: A plain English description of the query
    """
    if not sql or not isinstance(sql, str):
        return "Couldn't translate the query"
        
    sql = sql.strip().lower()
    
    try:
        # Extract main parts of the query
        select_part = re.search(r'select\s+(.*?)\s+from', sql, re.IGNORECASE | re.DOTALL)
        from_part = re.search(r'from\s+(.*?)(?:\s+where|\s+group by|\s+order by|\s+limit|\s*$)', sql, re.IGNORECASE | re.DOTALL)
        where_part = re.search(r'where\s+(.*?)(?:\s+group by|\s+order by|\s+limit|\s*$)', sql, re.IGNORECASE | re.DOTALL)
        group_by_part = re.search(r'group by\s+(.*?)(?:\s+having|\s+order by|\s+limit|\s*$)', sql, re.IGNORECASE | re.DOTALL)
        having_part = re.search(r'having\s+(.*?)(?:\s+order by|\s+limit|\s*$)', sql, re.IGNORECASE | re.DOTALL)
        order_by_part = re.search(r'order by\s+(.*?)(?:\s+limit|\s*$)', sql, re.IGNORECASE | re.DOTALL)
        limit_part = re.search(r'limit\s+(.*?)\s*$', sql, re.IGNORECASE | re.DOTALL)
        
        # Build the English description
        description = "I'm going to "
        
        # SELECT part
        if select_part:
            columns = select_part.group(1).strip()
            if "distinct" in columns:
                description += "find unique values of "
                columns = columns.replace("distinct", "").strip()
            else:
                description += "retrieve "
                
            # Check for aggregates
            if any(agg in columns.lower() for agg in ["sum(", "avg(", "count(", "min(", "max("]):
                for agg in ["sum(", "avg(", "count(", "min(", "max("]:
                    if agg in columns.lower():
                        agg_func = agg.replace("(", "")
                        description = description.replace("retrieve", f"calculate the {agg_func} of")
                        break
            
            # Clean up column names for better readability
            columns = columns.replace("as", "renamed to")
            description += columns
        
        # FROM part
        if from_part:
            tables = from_part.group(1).strip()
            description += f" from the {tables} data"
        
        # WHERE part
        if where_part:
            conditions = where_part.group(1).strip()
            description += f" where {conditions}"
        
        # GROUP BY part
        if group_by_part:
            group_cols = group_by_part.group(1).strip()
            description += f" grouped by {group_cols}"
        
        # HAVING part
        if having_part:
            having_conds = having_part.group(1).strip()
            description += f" having {having_conds}"
        
        # ORDER BY part
        if order_by_part:
            order_cols = order_by_part.group(1).strip()
            if "desc" in order_cols.lower():
                order_cols = order_cols.replace("desc", "in descending order")
            elif "asc" in order_cols.lower():
                order_cols = order_cols.replace("asc", "in ascending order")
            description += f" sorted by {order_cols}"
        
        # LIMIT part
        if limit_part:
            limit_val = limit_part.group(1).strip()
            description += f" showing only {limit_val} results"
        
        return description
    except Exception as e:
        return f"I'm querying the database (couldn't translate SQL: {e})"
