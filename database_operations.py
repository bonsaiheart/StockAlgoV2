from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.ddl import DropSchema, CreateSchema

from UTILITIES.logger_config import logger
from db_schema_models import Symbol,SymbolQuote, Option, OptionQuote, ProcessedOptionData, TechnicalAnalysis

def create_database_tables(engine):
    with engine.connect() as conn:  # Use a synchronous context manager
        inspector = inspect(conn)
        existing_tables = inspector.get_table_names()
        tables_to_create = [Symbol, Option, OptionQuote, SymbolQuote, TechnicalAnalysis, ProcessedOptionData]

        try:  # Wrap table creation in a try-except block
            # Create Symbol table first
            if Symbol.__table__.name not in existing_tables:
                Symbol.__table__.create(bind=engine)  # Use bind=engine to specify the engine

            # Then create the other tables
            for table in tables_to_create:
                if table.__table__.name != Symbol.__table__.name and table.__table__.name not in existing_tables:
                    table.__table__.create(bind=engine)

            logger.info("Database tables created or already exist.")
        except OperationalError as e:  # Catch OperationalError specifically
            logger.error(f"Error creating tables: {e}")
# Name of the new schema
NEW_SCHEMA = "public"

def drop_schema_if_exists(engine):
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Check if the schema exists
        inspector = inspect(engine)
        if NEW_SCHEMA in inspector.get_schema_names():
            # Drop the schema and all contained objects
            session.execute(DropSchema(NEW_SCHEMA, cascade=True))
            session.commit()
            logger.info(f"Schema '{NEW_SCHEMA}' dropped successfully.")
        else:
            logger.info(f"Schema '{NEW_SCHEMA}' does not exist. No need to drop.")
    except Exception as e:
        logger.error(f"Error dropping schema: {e}")
        session.rollback()
    finally:
        session.close()

def drop_create_schema_and_tables(engine):
    # First, drop the schema if it exists
    drop_schema_if_exists(engine)

    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Create the new schema
        session.execute(CreateSchema(NEW_SCHEMA))
        session.commit()
        logger.info(f"Schema '{NEW_SCHEMA}' created successfully.")
    except ProgrammingError:
        logger.info(f"Schema '{NEW_SCHEMA}' already exists.")
        session.rollback()

    # Modify the tables to use the new schema
    tables_to_create = [Symbol, Option, OptionQuote, SymbolQuote, TechnicalAnalysis, ProcessedOptionData]
    for table in tables_to_create:
        table.__table__.schema = NEW_SCHEMA

    # Create tables in the new schema
    with engine.connect() as conn:
        inspector = inspect(conn)
        existing_tables = inspector.get_table_names(schema=NEW_SCHEMA)

        try:
            for table in tables_to_create:
                if table.__table__.name not in existing_tables:
                    table.__table__.create(bind=engine)
                    logger.info(f"Table '{table.__table__.name}' created in schema '{NEW_SCHEMA}'.")
                else:
                    logger.info(f"Table '{table.__table__.name}' already exists in schema '{NEW_SCHEMA}'.")

            logger.info("All tables created or already exist in the new schema.")
        except OperationalError as e:
            logger.error(f"Error creating tables: {e}")

    session.close()