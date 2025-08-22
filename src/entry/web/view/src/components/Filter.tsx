import type { DropdownProps, EditableProps } from "@/types";

export function Dropdown({ name, options = [] }: DropdownProps) {
  return (
    <div className="flex flex-row items-center bg-white border-2 rounded-xl p-1">
      <div className="p-2">
        <label className="font-bold" htmlFor={name}>
          {name + " "}
        </label>
      </div>
      <div className="border rounded-md border-gray-300 p-1 px-2">
        <select id={name} name={name} className="focus:outline-none">
          {options.length === 0 ? (
            <option value="">No options available</option>
          ) : (
            options.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))
          )}
        </select>
      </div>
    </div>
  );
}

export function Editable({ name, defaultValue }: EditableProps) {
  return (
    <div className="flex flex-row items-center bg-white border-2 rounded-xl p-1">
      <div className="p-2">
        <label className="font-bold" htmlFor={name}>
          {name + " "}
        </label>
      </div>
      <div className="border rounded-md border-gray-300 p-1 px-2">
        <input
          id={name}
          name={name}
          defaultValue={defaultValue}
          size={5}
          className="min-w-0 w-fit focus:outline-none"
          placeholder="Enter value"
        />
      </div>
    </div>
  );
}

export function Checkbox({ name, defaultValue = false, label }: { name: string; defaultValue?: boolean; label?: string }) {
  return (
    <div className="flex flex-row items-center bg-white border-2 rounded-xl p-3 space-x-3">
      <label className="flex items-center space-x-2 cursor-pointer" htmlFor={name}>
        <input
          id={name}
          name={name}
          type="checkbox"
          defaultChecked={defaultValue}
          className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2 transition-all duration-200"
        />
        <span className="font-medium text-gray-700 select-none">
          {label || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
        </span>
      </label>
    </div>
  );
}
