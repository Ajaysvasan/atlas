from numpy import uint32


class InvalidFileType(Exception):
    def __init__(self, file_extention) -> None:
        self.file_extention = file_extention
        super().__init__(file_extention)

    def __str__(self):
        return f"Unsupported file type : {self.file_extention}"


class VectorInsertionError(Exception):
    def __init__(self, vector_id):
        self.vector_id = vector_id
        super().__init__(self.vector_id)

    def __str__(self):
        return f"An Error occured while inserting the vector : {self.vector_id}"


class IndexDirectoryDoesNotExists(Exception):
    def __init__(self, directory_name) -> None:
        self.directory_name = directory_name
        super().__init__(self.directory_name)

    def __str__(self):
        return f"The directory with the following name doesn't exist: {self.directory_name}"


class InsertionError(Exception):
    def __init__(self, error, tableName, id) -> None:
        self.error = error
        self.message = tableName
        self.id = id
        super().__init__(self.error)

    def __str__(self):
        return f"{self.error}:Error occured while inserting values in the following table {self.message} , for the id : {self.id}"


class InvalidEmbeddingArgument(Exception):
    def __init__(self, error_message) -> None:
        self.error_message = error_message
        super().__init__(error_message)

    def __str__(self):
        return self.error_message


class VectorNotFoundEror(Exception):
    def __init__(self, vector_id: uint32) -> None:
        self.vector_id = vector_id
        super().__init__(vector_id)

    def __str__(self):
        return f"No vectors found for the vector Id : {self.vector_id}"


class DuplicateVectorException(Exception):

    def __init__(self, vector_id: uint32) -> None:
        self.vector_id = vector_id
        super().__init__(vector_id)

    def __str__(self):
        return f"The vector with vector id {self.vector_id} , already exists"


class InvalidBatchSize(Exception):

    def __init__(self, error_message) -> None:
        self.error_message = error_message
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message


class InvalidVectorDimension(Exception):
    def __init__(self, passed_dimension: int, expected_dimension: int) -> None:
        self.passed_dimension = passed_dimension
        self.expected_dimension = expected_dimension
        super().__init__(self.passed_dimension, self.expected_dimension)

    def __str__(self) -> str:
        return f"Expected dimension {self.expected_dimension} , got {self.passed_dimension}"
