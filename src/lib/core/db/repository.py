"""
Repository pattern implementation for database operations.

This module provides a clean abstraction layer over database operations,
implementing the repository pattern for consistent data access.
"""

from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel

from lib.core.db.base import Database
from lib.core.exceptions.data import DatabaseError, DatabaseQueryError, DataNotFoundError, DataValidationException
from lib.core.logging_config import get_logger

# Generic type for models
T = TypeVar("T", bound=BaseModel)
ID = TypeVar("ID", str, int, UUID)

# Module logger
_logger = get_logger("lib.core.db.repository")


class QueryFilter:
    """
    Query filter for repository operations.

    Provides a clean way to construct WHERE clauses for SQL queries.
    """

    def __init__(self):
        self._conditions = []
        self._params = []

    def add(self, field: str, operator: str, value: Any) -> "QueryFilter":
        """
        Add a condition to the filter.

        Args:
            field: Field name
            operator: Comparison operator (=, >, <, !=, LIKE, etc.)
            value: Value to compare against

        Returns:
            Self for method chaining
        """
        self._conditions.append(f"{field} {operator} %s")
        self._params.append(value)
        return self

    def add_raw(self, condition: str, *params: Any) -> "QueryFilter":
        """
        Add a raw SQL condition with parameters.

        Args:
            condition: Raw SQL condition with %s placeholders
            *params: Values for the placeholders

        Returns:
            Self for method chaining
        """
        self._conditions.append(condition)
        self._params.extend(params)
        return self

    def add_in(self, field: str, values: list[Any]) -> "QueryFilter":
        """
        Add an IN condition to the filter.

        Args:
            field: Field name
            values: List of values to check against

        Returns:
            Self for method chaining
        """
        if not values:
            return self

        placeholders = ", ".join(["%s" for _ in values])
        self._conditions.append(f"{field} IN ({placeholders})")
        self._params.extend(values)
        return self

    def build(self) -> tuple[str, list[Any]]:
        """
        Build the WHERE clause and parameters.

        Returns:
            Tuple of (sql_clause, params)
        """
        if not self._conditions:
            return "", []

        where_clause = " AND ".join(self._conditions)
        return f"WHERE {where_clause}", self._params


class Pagination:
    """
    Pagination parameters for repository queries.
    """

    def __init__(self, page: int = 1, page_size: int = 50, order_by: str | None = None, order_dir: str = "ASC"):
        """
        Initialize pagination.

        Args:
            page: Page number (1-based)
            page_size: Number of items per page
            order_by: Field to order by
            order_dir: Order direction (ASC or DESC)
        """
        self.page = max(1, page)
        self.page_size = min(max(1, page_size), 1000)  # Prevent excessive page sizes
        self.order_by = order_by
        self.order_dir = order_dir.upper() if order_dir.upper() in ["ASC", "DESC"] else "ASC"

    def get_limit_offset(self) -> tuple[int, int]:
        """
        Get the LIMIT and OFFSET values for SQL.

        Returns:
            Tuple of (limit, offset)
        """
        return self.page_size, (self.page - 1) * self.page_size

    def build_order_clause(self) -> str:
        """
        Build the ORDER BY clause.

        Returns:
            SQL ORDER BY clause or empty string
        """
        if not self.order_by:
            return ""

        # Sanitize order_by to prevent SQL injection
        # Only allow alphanumeric and underscore characters
        safe_order_by = "".join(c for c in self.order_by if c.isalnum() or c == "_")
        if not safe_order_by:
            return ""

        return f"ORDER BY {safe_order_by} {self.order_dir}"


class PaginatedResult(Generic[T]):
    """
    Result of a paginated query.
    """

    def __init__(self, items: list[T], total: int, page: int, page_size: int):
        """
        Initialize paginated result.

        Args:
            items: List of items on the current page
            total: Total number of items across all pages
            page: Current page number
            page_size: Number of items per page
        """
        self.items = items
        self.total = total
        self.page = page
        self.page_size = page_size
        self.total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    def has_next(self) -> bool:
        """Check if there's a next page."""
        return self.page < self.total_pages

    def has_prev(self) -> bool:
        """Check if there's a previous page."""
        return self.page > 1

    def next_page(self) -> int:
        """Get the next page number or current if at the end."""
        return min(self.page + 1, self.total_pages) if self.total_pages > 0 else 1

    def prev_page(self) -> int:
        """Get the previous page number or 1 if at the beginning."""
        return max(self.page - 1, 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "items": [item.model_dump() if hasattr(item, "model_dump") else item.dict() for item in self.items],
            "pagination": {
                "total": self.total,
                "page": self.page,
                "page_size": self.page_size,
                "total_pages": self.total_pages,
                "has_next": self.has_next(),
                "has_prev": self.has_prev(),
            },
        }


class BaseRepository(Generic[T, ID]):
    """
    Base repository for data access operations.

    Provides a generic implementation of the repository pattern
    for consistent data access across the application.
    """

    def __init__(
        self,
        db: Database,
        model_cls: type[T],
        table_name: str | None = None,
        id_field: str = "id",
        timestamps: bool = True,
        created_at_field: str = "created_at",
        updated_at_field: str = "updated_at",
    ):
        """
        Initialize repository.

        Args:
            db: Database instance
            model_cls: Pydantic model class
            table_name: Database table name (defaults to model class name lowercase + 's')
            id_field: Name of the ID field
            timestamps: Whether to automatically manage timestamps
            created_at_field: Name of the created_at field
            updated_at_field: Name of the updated_at field
        """
        self.db = db
        self.model_cls = model_cls
        self.table_name = table_name or f"{model_cls.__name__.lower()}s"
        self.id_field = id_field
        self.timestamps = timestamps
        self.created_at_field = created_at_field
        self.updated_at_field = updated_at_field

        # Get model fields
        self.model_fields = set(model_cls.__annotations__.keys())

        _logger.debug("repository_initialized", table=self.table_name, model=model_cls.__name__)

    async def get_by_id(self, id: ID) -> T | None:
        """
        Get a record by ID.

        Args:
            id: The record ID

        Returns:
            Model instance or None if not found

        Raises:
            DatabaseError: If a database error occurs
        """
        try:
            async with self.db.transaction() as conn:
                query = f"SELECT * FROM {self.table_name} WHERE {self.id_field} = %s"
                cursor = conn.cursor()
                cursor.execute(query, (id,))
                row = cursor.fetchone()

                if row is None:
                    return None

                return self._row_to_model(dict(row))

        except Exception as e:
            _logger.error("get_by_id_failed", table=self.table_name, id=id, exc_info=True)
            raise DatabaseError(
                message=f"Failed to get {self.model_cls.__name__} with ID {id}", details={"id": id, "error": str(e)}
            ) from e

    async def find_one(self, filter: QueryFilter) -> T | None:
        """
        Find a single record matching the filter.

        Args:
            filter: Query filter

        Returns:
            Model instance or None if not found

        Raises:
            DatabaseError: If a database error occurs
        """
        try:
            where_clause, params = filter.build()

            async with self.db.transaction() as conn:
                query = f"SELECT * FROM {self.table_name} {where_clause} LIMIT 1"
                cursor = conn.cursor()
                cursor.execute(query, params)
                row = cursor.fetchone()

                if row is None:
                    return None

                return self._row_to_model(dict(row))

        except Exception as e:
            _logger.error("find_one_failed", table=self.table_name, exc_info=True)
            raise DatabaseQueryError(
                message=f"Failed to find {self.model_cls.__name__}", details={"error": str(e)}
            ) from e

    async def find_all(
        self, filter: QueryFilter | None = None, pagination: Pagination | None = None
    ) -> PaginatedResult[T]:
        """
        Find all records matching the filter with pagination.

        Args:
            filter: Query filter
            pagination: Pagination parameters

        Returns:
            Paginated result with model instances

        Raises:
            DatabaseError: If a database error occurs
        """
        try:
            where_clause, params = filter.build() if filter else ("", [])
            pagination = pagination or Pagination()
            limit, offset = pagination.get_limit_offset()
            order_clause = pagination.build_order_clause()

            async with self.db.transaction() as conn:
                # Get total count
                count_query = f"SELECT COUNT(*) FROM {self.table_name} {where_clause}"
                cursor = conn.cursor()
                cursor.execute(count_query, params)
                total = cursor.fetchone()[0]

                # Get paginated results
                query = f"""
                    SELECT * FROM {self.table_name}
                    {where_clause}
                    {order_clause}
                    LIMIT %s OFFSET %s
                """
                cursor = conn.cursor()
                cursor.execute(query, params + [limit, offset])
                rows = cursor.fetchall()

                items = [self._row_to_model(dict(row)) for row in rows]

                return PaginatedResult(items=items, total=total, page=pagination.page, page_size=pagination.page_size)

        except Exception as e:
            _logger.error("find_all_failed", table=self.table_name, exc_info=True)
            raise DatabaseQueryError(
                message=f"Failed to find {self.model_cls.__name__} records", details={"error": str(e)}
            ) from e

    async def create(self, data: T | dict[str, Any]) -> T:
        """
        Create a new record.

        Args:
            data: Model instance or dictionary of field values

        Returns:
            Created model instance

        Raises:
            DatabaseError: If a database error occurs
            DataValidationException: If validation fails
        """
        try:
            # Convert to dict if it's a model
            if isinstance(data, BaseModel):
                # Use model_dump() for Pydantic v2, fallback to dict() for v1
                _data: Any = data
                if hasattr(_data, "model_dump"):
                    data_dict = _data.model_dump(exclude_unset=True)
                else:
                    data_dict = _data.dict(exclude_unset=True)
            else:
                data_dict = data

            # Filter out fields not in model
            filtered_data = {k: v for k, v in data_dict.items() if k in self.model_fields}

            # Add timestamps if enabled
            if self.timestamps:
                now = datetime.utcnow()
                if self.created_at_field in self.model_fields:
                    filtered_data[self.created_at_field] = now
                if self.updated_at_field in self.model_fields:
                    filtered_data[self.updated_at_field] = now

            # Build query
            fields = list(filtered_data.keys())
            placeholders = ["%s"] * len(fields)
            field_str = ", ".join(fields)
            placeholder_str = ", ".join(placeholders)
            values = [filtered_data[field] for field in fields]

            async with self.db.transaction() as conn:
                query = f"""
                    INSERT INTO {self.table_name} ({field_str})
                    VALUES ({placeholder_str})
                    RETURNING *
                """
                cursor = conn.cursor()
                cursor.execute(query, values)
                row = cursor.fetchone()

                return self._row_to_model(dict(row))

        except Exception as e:
            _logger.error("create_failed", table=self.table_name, exc_info=True)
            if isinstance(e, (ValueError, TypeError)):
                raise DataValidationException(
                    message=f"Invalid data for {self.model_cls.__name__}", details={"error": str(e)}
                ) from e
            else:
                raise DatabaseError(
                    message=f"Failed to create {self.model_cls.__name__}", details={"error": str(e)}
                ) from e

    async def update(self, id: ID, data: T | dict[str, Any]) -> T:
        """
        Update an existing record.

        Args:
            id: Record ID
            data: Model instance or dictionary of field values

        Returns:
            Updated model instance

        Raises:
            DatabaseError: If a database error occurs
            DataNotFoundError: If the record is not found
            DataValidationException: If validation fails
        """
        try:
            # Check if record exists
            existing = await self.get_by_id(id)
            if existing is None:
                raise DataNotFoundError(
                    message=f"{self.model_cls.__name__} with ID {id} not found",
                    data_type=self.model_cls.__name__,
                    data_id=str(id),
                )

            # Convert to dict if it's a model
            if isinstance(data, BaseModel):
                # Use model_dump() for Pydantic v2, fallback to dict() for v1
                _data_u: Any = data
                if hasattr(_data_u, "model_dump"):
                    data_dict = _data_u.model_dump(exclude_unset=True)
                else:
                    data_dict = _data_u.dict(exclude_unset=True)
            else:
                data_dict = data

            # Filter out fields not in model and ID field
            filtered_data = {k: v for k, v in data_dict.items() if k in self.model_fields and k != self.id_field}

            # Add updated timestamp if enabled
            if self.timestamps and self.updated_at_field in self.model_fields:
                filtered_data[self.updated_at_field] = datetime.utcnow()

            # If no fields to update, return existing
            if not filtered_data:
                return existing

            # Build query
            set_clauses = [f"{field} = %s" for field in filtered_data]
            set_clause = ", ".join(set_clauses)
            values = list(filtered_data.values())
            values.append(id)  # Add ID for WHERE clause

            async with self.db.transaction() as conn:
                query = f"""
                    UPDATE {self.table_name}
                    SET {set_clause}
                    WHERE {self.id_field} = %s
                    RETURNING *
                """
                cursor = conn.cursor()
                cursor.execute(query, values)
                row = cursor.fetchone()

                if row is None:
                    raise DataNotFoundError(
                        message=f"{self.model_cls.__name__} with ID {id} not found",
                        data_type=self.model_cls.__name__,
                        data_id=str(id),
                    )

                return self._row_to_model(dict(row))

        except DataNotFoundError:
            # Re-raise not found errors
            raise
        except Exception as e:
            _logger.error("update_failed", table=self.table_name, id=id, exc_info=True)
            if isinstance(e, (ValueError, TypeError)):
                raise DataValidationException(
                    message=f"Invalid data for {self.model_cls.__name__}", details={"error": str(e)}
                ) from e
            else:
                raise DatabaseError(
                    message=f"Failed to update {self.model_cls.__name__} with ID {id}",
                    details={"id": id, "error": str(e)},
                ) from e

    async def delete(self, id: ID) -> bool:
        """
        Delete a record by ID.

        Args:
            id: Record ID

        Returns:
            True if the record was deleted, False if not found

        Raises:
            DatabaseError: If a database error occurs
        """
        try:
            async with self.db.transaction() as conn:
                query = f"DELETE FROM {self.table_name} WHERE {self.id_field} = %s RETURNING {self.id_field}"
                cursor = conn.cursor()
                cursor.execute(query, (id,))
                result = cursor.fetchone()

                return result is not None

        except Exception as e:
            _logger.error("delete_failed", table=self.table_name, id=id, exc_info=True)
            raise DatabaseError(
                message=f"Failed to delete {self.model_cls.__name__} with ID {id}", details={"id": id, "error": str(e)}
            ) from e

    async def count(self, filter: QueryFilter | None = None) -> int:
        """
        Count records matching the filter.

        Args:
            filter: Query filter

        Returns:
            Number of matching records

        Raises:
            DatabaseError: If a database error occurs
        """
        try:
            where_clause, params = filter.build() if filter else ("", [])

            async with self.db.transaction() as conn:
                query = f"SELECT COUNT(*) FROM {self.table_name} {where_clause}"
                cursor = conn.cursor()
                cursor.execute(query, params)
                result = cursor.fetchone()

                return result[0] if result else 0

        except Exception as e:
            _logger.error("count_failed", table=self.table_name, exc_info=True)
            raise DatabaseQueryError(
                message=f"Failed to count {self.model_cls.__name__} records", details={"error": str(e)}
            ) from e

    async def exists(self, id: ID) -> bool:
        """
        Check if a record exists by ID.

        Args:
            id: Record ID

        Returns:
            True if the record exists, False otherwise

        Raises:
            DatabaseError: If a database error occurs
        """
        try:
            async with self.db.transaction() as conn:
                query = f"SELECT 1 FROM {self.table_name} WHERE {self.id_field} = %s LIMIT 1"
                cursor = conn.cursor()
                cursor.execute(query, (id,))
                result = cursor.fetchone()

                return result is not None

        except Exception as e:
            _logger.error("exists_check_failed", table=self.table_name, id=id, exc_info=True)
            raise DatabaseQueryError(
                message=f"Failed to check if {self.model_cls.__name__} exists with ID {id}",
                details={"id": id, "error": str(e)},
            ) from e

    async def execute_raw(self, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        """
        Execute a raw SQL query.

        This method is for advanced use cases where the repository
        methods don't provide enough flexibility.

        Args:
            query: Raw SQL query
            params: Query parameters

        Returns:
            List of result rows as dictionaries

        Raises:
            DatabaseError: If a database error occurs
        """
        try:
            params = params or []

            async with self.db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)

                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row, strict=False)) for row in rows]

                return []

        except Exception as e:
            _logger.error("raw_query_failed", table=self.table_name, exc_info=True)
            raise DatabaseQueryError(
                message="Failed to execute raw SQL query", details={"error": str(e)}, sql=query
            ) from e

    def _row_to_model(self, row: dict[str, Any]) -> T:
        """
        Convert a database row to a model instance.

        Args:
            row: Database row as dictionary

        Returns:
            Model instance
        """
        try:
            # Filter out fields not in model
            filtered_data = {k: v for k, v in row.items() if k in self.model_fields}
            return self.model_cls(**filtered_data)
        except Exception as e:
            _logger.error("row_to_model_failed", table=self.table_name, model=self.model_cls.__name__, exc_info=True)
            raise DataValidationException(
                message=f"Failed to convert database row to {self.model_cls.__name__}",
                details={"error": str(e), "row": row},
            ) from e
