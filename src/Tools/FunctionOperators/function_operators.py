import ast
import random
import astor


__all__ = [
    "perturb_constants",
    "flip_if_condition",
    "insert_random_return",
    "merge_adjacent_if",
    "crossover_functions",
]


def perturb_constants(
    code: str, 
    perturb_factor: float = 0.3
) -> str:
    
    """
    遍历代码中的常数，将每个常数值上下扰动一定的比例（默认30%）。
    
    参数:
        code (str): 输入的 Python 代码字符串。
        perturb_factor (float): 扰动因子的比例，默认为 0.3（即±30%）。
    
    返回值:
        str: 修改后的 Python 代码字符串。
    """
    
    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            perturbation = random.uniform(1 - perturb_factor, 1 + perturb_factor)
            node.value = node.value * perturbation
    
    return astor.to_source(tree)


def insert_random_return(
    code: str, 
    return_value: "int | float | str | None" = 0.0,
    drop_return_prob: float = 0.3, 
)-> str:
    
    """
    在函数内部随机插入一个 `return` 语句，插入一个 `return` 后任务即完成，不继续插入其他 `return`。
    
    自动根据给定的 return_value 类型生成 `return` 语句，支持 `int`、`float`、`str`、`None`。
    
    参数:
        code (str): 输入的 Python 代码字符串。
        return_value (int | float | str | None): `return` 语句的返回值，默认为 0.0。
        drop_return_prob (float): 遍历语法树时在某一位置插入 `return` 语句的概率，默认为 0.3。

    返回值:
        str: 修改后的 Python 代码字符串。
    """
    
    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and random.random() < drop_return_prob:
            if return_value is None:
                return_stmt = ast.Return(value=ast.Constant(value=None))
            elif isinstance(return_value, (int, float, str)):
                return_stmt = ast.Return(value=ast.Constant(value=return_value))
            else:
                raise ValueError(f"Unsupported return_value type: {type(return_value)}")
            
            pos = random.randint(0, len(node.body))
            node.body.insert(pos, return_stmt)
            break
    
    return astor.to_source(tree)


def merge_adjacent_if(
    code: str
)-> str:
    
    """
    合并相邻的 `if` 语句：无论条件是否相同，以 50% 概率用 `and` 或 `or` 连接条件。

    参数:
        code (str): 输入的 Python 代码字符串。

    返回值:
        str: 修改后的 Python 代码字符串（只做一次合并）。
    """
    
    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.orelse, list) and len(node.orelse) == 1:
            next_if = node.orelse[0]
            if isinstance(next_if, ast.If):
                op = ast.And() if random.random() < 0.5 else ast.Or()
                new_test = ast.BoolOp(op=op, values=[node.test, next_if.test])
                node.test = new_test
                node.body.extend(next_if.body)
                node.orelse = next_if.orelse
                return astor.to_source(tree)

    return astor.to_source(tree)


def crossover_functions(
    code1: str,
    code2: str
) -> str:
    """
    将两个函数的代码交叉组合，要求两函数签名完全相同。参数名保持不变，
    函数体按顺序随机交错组合，遇到 return 截断。

    参数:
        code1 (str): 第一个函数的代码字符串。
        code2 (str): 第二个函数的代码字符串。

    返回:
        str: 合成后的新函数代码字符串。
    """

    def parse_and_get_function(code: str) -> ast.FunctionDef:
        mod = ast.parse(code)
        for node in mod.body:
            if isinstance(node, ast.FunctionDef):
                return node
        raise ValueError("代码中未找到函数定义。")

    func1 = parse_and_get_function(code1)
    func2 = parse_and_get_function(code2)

    body1 = func1.body
    body2 = func2.body

    i, j = 0, 0
    mixed_body = []
    done1 = done2 = False

    while not (done1 and done2):
        choose_first = (not done1 and (done2 or random.random() < 0.5))
        if choose_first and i < len(body1):
            stmt = body1[i]
            mixed_body.append(stmt)
            i += 1
            if isinstance(stmt, ast.Return): break
        elif not choose_first and j < len(body2):
            stmt = body2[j]
            mixed_body.append(stmt)
            j += 1
            if isinstance(stmt, ast.Return): break
        if i >= len(body1): done1 = True
        if j >= len(body2): done2 = True

    new_func = ast.FunctionDef(
        name=func1.name,
        args=func1.args,
        body=mixed_body,
        decorator_list=[],
        returns=func1.returns,
        type_comment=func1.type_comment,
        lineno=1
    )

    mod = ast.Module(body=[new_func], type_ignores=[])
    return ast.unparse(mod)


if __name__ == "__main__":
    
    code1 = r"""def priority(
    current_point: tuple[int, int, int, int, int, int, int, int],
    history_points: list[tuple[int, int, int, int, int, int, int, int]],
) -> float:
    current = np.array(current_point)
    set_size = len(history_points)
    
    # Phase detection with smooth transitions
    early_phase = 1.0 / (1.0 + np.exp((set_size - 150) / 30.0))
    mid_phase = 1.0 / (1.0 + np.exp(-(set_size - 350) / 70.0))
    late_phase = 1.0 / (1.0 + np.exp(-(set_size - 500) / 60.0))
    
    # Feature 1: Multi-resolution AP detection
    ap_penalty = 0.0
    if history_points:
        history = np.array(history_points)
        # Check all possible 2-point combinations
        for scale in [1, 2, 3]:
            for dim_group in [[i] for i in range(8)] + [list(range(i, i+scale)) for i in range(8-scale+1)]:
                current_proj = current[dim_group]
                hist_proj = history[:, dim_group]
                # Find potential midpoints that would complete APs
                mid_candidates = (hist_proj + current_proj) * 2 % 3
                # Vectorized comparison
                matches = np.any(np.all(mid_candidates[:, np.newaxis] == hist_proj, axis=2), axis=1)
                penalty = np.sum(matches) * (1.0 + 0.3 * len(dim_group))
                ap_penalty -= penalty
        ap_penalty = max(-20.0, ap_penalty) / 15.0  # Normalized
    
    # Feature 2: Dynamic dimension weighting
    dim_weights = np.ones(8)
    if history_points:
        # Compute dimension-wise value distribution
        dim_variance = np.zeros(8)
        for d in range(8):
            counts = np.bincount(history[:, d], minlength=3)
            dim_variance[d] = np.var(counts)
        # Prefer dimensions with balanced distribution
        dim_weights = 1.5 - dim_variance / np.max(dim_variance)
    
    # Feature 3: Component diversity
    component_counts = np.bincount(current, minlength=3)
    diversity_score = 1.0 - (np.max(component_counts) - np.min(component_counts)) / 6.0
    
    # Feature 4: Spatial anti-clustering
    spatial_score = 1.0
    if history_points:
        # Compute weighted distance (emphasizing certain dimensions)
        weighted_diff = (history - current) * dim_weights
        distances = np.sum(weighted_diff**2, axis=1)
        spatial_score = np.mean(distances) / 20.0
    
    # Feature 5: Pattern uniqueness
    pattern_score = 1.0
    if history_points:
        # Count occurrences of current point's value pattern
        pattern_matches = np.sum(np.all(history == current, axis=1))
        pattern_score = 1.0 / (1.0 + pattern_matches)
    
    # Dynamic feature fusion
    weights = np.array([
        3.5 * (1 - early_phase) + 2.0 * early_phase,  # ap_penalty
        1.8 * (1 - mid_phase) + 1.2 * mid_phase,      # dim_weights
        1.5 * (1 - late_phase) + 2.0 * late_phase,    # diversity_score
        1.2 * early_phase + 0.8 * (1 - early_phase),  # spatial_score
        0.7 * (1 - mid_phase) + 0.3 * mid_phase       # pattern_score
    ])
    
    priority_score = (
        weights[0] * (1.0 + ap_penalty) +  # ap_penalty is negative
        weights[1] * np.mean(dim_weights) +
        weights[2] * diversity_score +
        weights[3] * spatial_score +
        weights[4] * pattern_score
    )
    
    # Late-phase adjustments
    if late_phase > 0.7:
        # Bonus for points with excellent dimension weights
        if np.min(dim_weights) > 1.3:
            priority_score *= 1.15
        # Penalty for points with component imbalance
        if np.max(component_counts) - np.min(component_counts) > 3:
            priority_score *= 0.9
    
    return float(priority_score)
"""

    code2 = r"""def priority(
    current_point: tuple[int, int, int, int, int, int, int, int],
    history_points: list[tuple[int, int, int, int, int, int, int, int]],
) -> float:

    current = np.array(current_point)
    set_size = len(history_points)
    
    # Phase calculation with exponential decay
    early_phase = np.exp(-set_size / 150.0)
    mid_phase = np.exp(-((set_size - 300) / 200.0)**2)
    late_phase = np.exp(-set_size / 600.0)
    
    # Feature 1: Fourier pattern analysis
    fourier_score = 0.0
    if history_points:
        history = np.array(history_points)
        # Compute 3D Fourier transform in selected subspaces
        for dims in [(0,1,2), (3,4,5), (1,3,5), (2,4,6)]:
            subspace = history[:, dims]
            # Compute discrete Fourier transform
            freq = np.fft.fftn(subspace, axes=(0,1,2))
            power = np.abs(freq)
            # Penalize high power at non-zero frequencies
            fourier_score -= np.sum(power[1:,1:,1:]) / (subspace.shape[0] * 8)
        fourier_score = max(-10.0, fourier_score) / 3.0  # Normalize
    
    # Feature 2: Hyper-spherical dispersion
    if history_points:
        # Convert to spherical coordinates in 8D
        norms = np.linalg.norm(history, axis=1)
        unit_vectors = history / norms[:, np.newaxis]
        current_norm = np.linalg.norm(current)
        current_unit = current / current_norm
        
        # Compute angular distances
        dots = np.dot(unit_vectors, current_unit)
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        min_angle = np.min(angles)
        dispersion_score = min_angle / np.pi
    else:
        dispersion_score = 1.0
    
    # Feature 3: Entropy maximization with component balancing
    counts = np.bincount(current, minlength=3)
    probs = counts / 8.0
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    balance_score = entropy / np.log(3)
    
    # Feature 4: Anti-diagonal correlation
    anti_corr = 0.0
    if set_size > 20:
        last_20 = history[-20:]
        for dim in range(8):
            opp_dim = 7 - dim
            same_count = sum(1 for p in last_20 if p[dim] == current[opp_dim])
            anti_corr += same_count / 20.0
        anti_corr = 1.0 - (anti_corr / 8.0)
    else:
        anti_corr = 1.0
    
    # Dynamic feature weighting
    weights = np.array([
        2.0 * (1 - early_phase) + 1.0 * early_phase,  # fourier_score
        1.5 * mid_phase + 2.0 * (1 - mid_phase),      # dispersion_score
        1.8 * (1 - late_phase) + 1.2 * late_phase,    # balance_score
        0.8 * (1 + early_phase)                       # anti_corr
    ])
    
    # Compose final priority score
    priority_score = (
        weights[0] * (1.0 + fourier_score) +  # fourier_score is negative
        weights[1] * dispersion_score +
        weights[2] * balance_score +
        weights[3] * anti_corr
    )
    
    # Late-phase adjustments
    if late_phase < 0.4:
        # Reward excellent component balance
        if np.max(counts) - np.min(counts) <= 1:
            priority_score *= 1.4
        # Penalize simple repeating patterns
        if len(set(current)) < 3:
            priority_score *= 0.7
    
    return float(priority_score)
"""

    new_code = crossover_functions(
        code1 = code1,
        code2 = code2,
    )
    
    print(new_code)