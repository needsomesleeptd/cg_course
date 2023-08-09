//
// Created by Андрей on 09.08.2023.
//

#ifndef DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_BASE_MATH_EXCEPTION_H_
#define DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_BASE_MATH_EXCEPTION_H_

#include <cstdio>
#include <exception>
const int DEFAULT_EXC_STR_LEN = 400;

class BaseException : public std::exception
{
 public:
	explicit BaseException(
		const char* filename,
		const int line,
		const char* className,
		const char* time,
		const char* exceptionName,
		const char* additionalInfo)
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

#endif //DZ2_CG_COURSE_PROGRAMM_MATH_PRIMITIVES_BASE_MATH_EXCEPTION_H_
