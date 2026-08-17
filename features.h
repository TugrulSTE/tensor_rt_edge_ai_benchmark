#pragma once
#include <iostream>
#include <map>
#include <string>

// It can be customized according to the specific trained model classes.
const std::map<int, std::string> class_list = {
	{1, "LIMIT 100" },
	{2, "LIMIT 120"},
	{3, "LIMIT 30"},
	{4, "LIMIT 50"},
	{5, "LIMIT 70"},
	{6, "LIMIT 80"},
	{7, "NO ENTRY"},
	{8, "NO PASS"},
	{9, "TRAFFIC SIGNAL"},
	{10, "STOP"},
	{11, "YIELD"}
};

const unsigned int raw_features[] = {
/* TODO: This field can be filled using Edge Impulse Raw Data Feature with the format of "0xAABBCC" */
};
