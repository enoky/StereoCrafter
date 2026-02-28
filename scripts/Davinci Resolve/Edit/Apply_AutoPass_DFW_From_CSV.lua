-- Apply_AutoPass_DFW_From_CSV_StepOnly.lua
-- Reads auto_pass_export.csv and keyframes Floating_Window.LeftDFW / RightDFW
-- Start-keys only (assumes Step/hold interpolation).

local comp = fu:GetCurrentComp()
if not comp then
    print("No active comp.")
    return
end

-- ---- EDIT THESE IF YOUR NODE NAMES DIFFER ----
local TOOL_NAME         = "Floating_Window"  -- your sMerge node name
local LEFT_INPUT_NAME   = "LeftDFW"          -- user control ID
local RIGHT_INPUT_NAME  = "RightDFW"         -- user control ID
local FRAME_OFFSET      = 0                  -- set if CSV frame 0 != comp frame 0
local VALUE_CLAMP_MIN   = 0.0
local VALUE_CLAMP_MAX   = 5.0                -- adjust if you ever allow >5
local CHANGE_EPS        = 1e-6               -- tolerance for "same value"
-- --------------------------------------------

local function clamp(x, a, b)
    if x < a then return a end
    if x > b then return b end
    return x
end

local function trim(s)
    return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

-- Robust CSV parser (handles quoted fields + commas)
local function parseCSVLine(line)
    local out = {}
    local i, len = 1, #line
    local field = ""
    local inQuotes = false

    while i <= len do
        local c = line:sub(i,i)
        if inQuotes then
            if c == '"' then
                local nxt = line:sub(i+1,i+1)
                if nxt == '"' then
                    field = field .. '"'
                    i = i + 1
                else
                    inQuotes = false
                end
            else
                field = field .. c
            end
        else
            if c == '"' then
                inQuotes = true
            elseif c == "," then
                table.insert(out, field)
                field = ""
            else
                field = field .. c
            end
        end
        i = i + 1
    end
    table.insert(out, field)
    return out
end

local function requestFile()
    return fu:RequestFile("Select auto_pass_export.csv")
end

local csvPath = requestFile()
if not csvPath then
    print("No CSV selected.")
    return
end

local tool = comp:FindTool(TOOL_NAME)
if not tool then
    print("Could not find tool: " .. tostring(TOOL_NAME))
    return
end

-- Ensure inputs are animatable splines
local function ensureSplineInput(t, inputName)
    local v = t[inputName]
    if v == nil or type(v) ~= "table" or v.ID ~= "BezierSpline" then
        t[inputName] = comp:BezierSpline()
    end
    return t[inputName]
end

local leftSpline  = ensureSplineInput(tool, LEFT_INPUT_NAME)
local rightSpline = ensureSplineInput(tool, RIGHT_INPUT_NAME)

-- Read CSV
local f = io.open(csvPath, "r")
if not f then
    print("Failed to open: " .. tostring(csvPath))
    return
end

local header = f:read("*l")
if not header then
    f:close()
    print("Empty CSV.")
    return
end

local cols = parseCSVLine(header)
local colIndex = {}
for i, c in ipairs(cols) do
    colIndex[trim(c)] = i
end

local iFrame = colIndex["frame"]
local iLeft  = colIndex["left_border"]
local iRight = colIndex["right_border"]

if not iFrame or not iLeft or not iRight then
    f:close()
    print("CSV missing required columns: frame, left_border, right_border")
    return
end

local rows = {}
for line in f:lines() do
    if line and line ~= "" then
        local parts = parseCSVLine(line)
        local fr = tonumber(trim(parts[iFrame] or ""))
        local l  = tonumber(trim(parts[iLeft]  or ""))
        local r  = tonumber(trim(parts[iRight] or ""))
        if fr and l and r then
            table.insert(rows, {frame = fr, left = l, right = r})
        end
    end
end
f:close()

if #rows == 0 then
    print("No usable rows found.")
    return
end

table.sort(rows, function(a,b) return a.frame < b.frame end)

-- Collapse into segments (only when values change)
local segments = {}
do
    local cur = rows[1]
    table.insert(segments, {frame=cur.frame, left=cur.left, right=cur.right})

    for i = 2, #rows do
        local rr = rows[i]
        local prev = segments[#segments]
        if math.abs(rr.left - prev.left) > CHANGE_EPS or math.abs(rr.right - prev.right) > CHANGE_EPS then
            table.insert(segments, {frame=rr.frame, left=rr.left, right=rr.right})
        end
    end
end

-- Write START-only keys (assumes stepped interpolation)
local written = 0
for i = 1, #segments do
    local seg = segments[i]
    local startF = seg.frame + FRAME_OFFSET

    local l = clamp(seg.left,  VALUE_CLAMP_MIN, VALUE_CLAMP_MAX)
    local r = clamp(seg.right, VALUE_CLAMP_MIN, VALUE_CLAMP_MAX)

    leftSpline[startF]  = l
    rightSpline[startF] = r
    written = written + 1
end

print(string.format(
    "Done. Segments=%d, keyframes written=%d (start-only) on %s.%s / %s.%s (offset=%d)",
    #segments, written, TOOL_NAME, LEFT_INPUT_NAME, TOOL_NAME, RIGHT_INPUT_NAME, FRAME_OFFSET
))