//
// Created by Андрей on 09.08.2023.
//

#ifndef DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_VECTOR_VECTOR_EXCEPTIONS_H_
#define DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_VECTOR_VECTOR_EXCEPTIONS_H_

#include "base_math_exception.h"

class InvalidMallocException : public BaseException
{
 public:
	InvalidMallocException(
		const char* filename,
		const int line,
		const char* className,
		const char* time,
		const char* exceptionName = "InvalidMallocException",
		const char* additionalInfo = "requested creation of invalid vector"
	) : BaseException(filename, line, className, time, exceptionName, additionalInfo)
	{
		sprintf(errorText,
			"In file: %s\n"
			"at line: %d\n"
			"in class: %s\n"
			"on time: %s\n"
			"source: %s\n"
			"reason: %s\n",
			filename, line, className, time, exceptionName, additionalInfo);
	}

	virtual const char* what() const noexcept
	{
		return errorText;
	}

 private:
	char errorText[DEFAULT_EXC_STR_LEN];
};

class InvalidOperationException : public BaseException
{
 public:
	InvalidOperationException(
		const char* filename,
		const int line,
		const char* className,
		const char* time,
		const char* exceptionName = "InvalidOperationException",
		const char* additionalInfo = "requested operation is not valid for given primitives"
	) : BaseException(filename, line, className, time, exceptionName, additionalInfo)
	{
		sprintf(errorText,
			"In file: %s\n"
			"at line: %d\n"
			"in class: %s\n"
			"on time: %s\n"
			"source: %s\n"
			"reason: %s\n",
			filename, line, className, time, exceptionName, additionalInfo);
	}

	virtual const char* what() const noexcept
	{
		return errorText;
	}

 private:
	char errorText[DEFAULT_EXC_STR_LEN];
};
#endif //DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_VECTOR_VECTOR_EXCEPTIONS_H_
